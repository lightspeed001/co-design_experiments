//! tpu_lowering — companion to post **1.2 "Lowering to the TPU"**.
//!
//! Same tile-level input as post 0.1 (`tile.gemm`), but this time the rewrite
//! emits a **TPU-shaped program** instead of scalar loops. The output is
//! deliberately *fatter* and *fewer*: HBM staging, VMEM scratchpad tiles, and
//! `tile.mxm` — one op per 128×128×128 systolic matmul, which is the shape the
//! MXU actually wants. That contrast is the whole point of the post:
//!
//! ```text
//! // ---- before: tile-level IR (unchanged from 0.1, bigger shapes) ----
//! "tile.gemm"(%a, %b, %c) {m = 256, n = 256, k = 256} : (…) -> ()
//!
//! // ---- after: lower_tile_gemm_to_tpu ----
//! // %a_v, %b_v, %c_v live in memory space 1 (VMEM) — the scratchpad
//! scf.for %ti = %c0 to %c2 step %c1 {            // 2×2 output tiles
//!   scf.for %tj = %c0 to %c2 step %c1 {
//!     "tile.zero"(%c_v)                          // vector unit fills C
//!     scf.for %tk = %c0 to %c2 step %c1 {        // K-dim streaming
//!       "tile.dma"(%a_hbm, %ti, %tk, %a_v) {direction = "hbm_to_vmem"}
//!       "tile.dma"(%b_hbm, %tk, %tj, %b_v) {direction = "hbm_to_vmem"}
//!       "tile.mxm"(%c_v, %a_v, %b_v)             // <- the MXU "instruction"
//!     }
//!     "tile.dma"(%c_v, %ti, %tj, %c_hbm) {direction = "vmem_to_hbm"}
//!   }
//! }
//! ```
//!
//! Production notes (all discussed in the post):
//! * Real Mosaic/TPU code uses `tpu_dma_start/wait` with semaphores to
//!   *double-buffer* the K loop; `tile.dma` here is the readable stand-in.
//! * The inner tile GEMM becomes an MXU-mapped op (`tile.mxm`), not loops —
//!   scalar loops would strand the systolic array.
//! * Layouts (bf16 tiles are (8 sublane × 128 lane) in Mosaic), fusion of
//!   epilogues onto the vector unit, and pipeline schedules are the parts our
//!   portable `tile.*` IR can't express — the "leak" row of the scorecard.
//!
//! `cargo run --bin tpu_lowering` prints before/after and runs upstream passes.

use std::error::Error;

use melior::{
    dialect::{func, DialectRegistry},
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, Type},
        *,
    },
    pass::{transform, PassManager},
    utility::register_all_dialects,
    Context,
};

/// The GEMM, in MXU-friendly shapes: tile counts are 2×2×2 of 128×128×128.
/// (Real workloads are much larger; what matters is that M, N, K divide by 128 —
/// the first perf pitfall in every TPU post.)
const M: i64 = 256;
const N: i64 = 256;
const K: i64 = 256;
const TILE_M: i64 = 128;
const TILE_N: i64 = 128;
const TILE_K: i64 = 128;

/// Memory spaces, in the numbering jax/Mosaic-style stacks use. Encoded in the
/// memref type (`memref<128x128xf32, 1 : i32>`), because *where a buffer lives*
/// is part of the type on TPU-flavored IR — one of the biggest "leaks" between
/// portable IR and target IR.
const HBM: &str = "0 : i32"; // off-chip HBM — big, slow-ish (TB/s)
const VMEM: &str = "1 : i32"; // on-chip scratchpad — small, very fast (tens of TB/s)

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    // tile.{gemm,zero,dma,mxm} are our stand-in dialect (see post 0.1).
    context.set_allow_unregistered_dialects(true);

    // ---------------------------------------------------------------
    // Phase 1 — the tile-level input. Identical contract to 0.1; only
    // the shapes grew so the tiling story becomes visible.
    // ---------------------------------------------------------------
    let before = build_tile_level(&context)?;
    println!("// ---- before: tile-level IR (portable middle) ----");
    println!("{}", before.as_operation());
    assert!(before.as_operation().verify(), "tile-level module must verify");

    // ---------------------------------------------------------------
    // Phase 2 — the TPU-shaped lowering.
    // ---------------------------------------------------------------
    let mut after = build_tpu_shaped(&context)?;
    println!("\n// ---- after: lower_tile_gemm_to_tpu ----");
    println!("{}", after.as_operation());

    // ---------------------------------------------------------------
    // Phase 3 — upstream passes from Rust (same machinery as 0.1).
    // ---------------------------------------------------------------
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(transform::create_canonicalizer_pass());
    pass_manager.add_pass(transform::create_cse_pass());
    pass_manager.run(&mut after)?;
    println!("\n// ---- after: canonicalize + cse ----");
    println!("{}", after.as_operation());
    assert!(after.as_operation().verify(), "TPU-shaped module must verify");

    Ok(())
}

/// memref type with an explicit memory space: `memref<128x128xf32, 1 : i32>`.
/// The space is part of the *type* — buffers that will meet on the same DMA
/// must agree about where they live.
fn memref(context: &Context, rows: i64, cols: i64, space: &str) -> Result<Type, Box<dyn Error>> {
    Ok(Type::parse(
        context,
        &format!("memref<{rows}x{cols}xf32, {space}>"),
    )?)
}

/// Push an `arith.constant <value> : index` and return its SSA value.
fn index_const(
    context: &Context,
    block: &Block<'_, '_>,
    location: Location,
    value: i64,
) -> Result<Value, Box<dyn Error>> {
    let op = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(
            Identifier::new(context, "value"),
            IntegerAttribute::new(value, Type::index(context)).into(),
        )])
        .add_results(&[Type::index(context)])
        .build()?;
    Ok(block.append_operation(op)?.result(0).unwrap().into())
}

/// Phase 1: `"tile.gemm"(%a, %b, %c) {m, n, k}` on HBM-resident memrefs.
fn build_tile_level(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let hbm = memref(context, M, N, HBM)?; // square shapes for readability
    let index_type = Type::index(context);

    let module = Module::new(location);
    let block = Block::new(&[(hbm, location), (hbm, location), (hbm, location)]);
    let a: Value = block.argument(0)?.into();
    let b: Value = block.argument(1)?.into();
    let c: Value = block.argument(2)?.into();

    block.append_operation(
        OperationBuilder::new("tile.gemm", location)
            .add_operands(&[a, b, c])
            .add_attributes(&[
                (Identifier::new(context, "m"), IntegerAttribute::new(M, index_type).into()),
                (Identifier::new(context, "n"), IntegerAttribute::new(N, index_type).into()),
                (Identifier::new(context, "k"), IntegerAttribute::new(K, index_type).into()),
            ])
            .build()?,
    )?;
    block.append_operation(func::r#return(&[], location))?;

    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "expert_gemm"),
        TypeAttribute::new(FunctionType::new(context, &[hbm, hbm, hbm], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}

/// Phase 2: the TPU-shaped rewrite.
///
/// Emission order mirrors how the hardware wants to run:
///   1. VMEM tile buffers (`memref.alloc` with memory space 1),
///   2. a 2×2 loop over output tiles,
///   3. zero the VMEM accumulator (vector-unit fill),
///   4. a K loop whose body is: DMA tiles in → one `tile.mxm` on the MXU,
///   5. DMA the result tile back to HBM,
///   6. an *epilogue slot* comment block — where bias/activation would fuse.
fn build_tpu_shaped(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let hbm = memref(context, M, N, HBM)?;
    let vmem = memref(context, TILE_M, TILE_N, VMEM)?;

    let module = Module::new(location);
    let block = Block::new(&[(hbm, location), (hbm, location), (hbm, location)]);
    let a_hbm: Value = block.argument(0)?.into();
    let b_hbm: Value = block.argument(1)?.into();
    let c_hbm: Value = block.argument(2)?.into();

    // Loop bounds: 2 tiles per dimension (256 / 128).
    let c0 = index_const(context, &block, location, 0)?;
    let c1 = index_const(context, &block, location, 1)?;
    let c_tiles = index_const(context, &block, location, M / TILE_M)?;

    // ---- VMEM scratchpad tiles. memref.alloc with a memory space is the
    // generic-MLIR way to say "this buffer lives on the scratchpad". A real
    // Mosaic program would also pick the (8, 128) lane/sublane layout; we keep
    // f32 here so the dump stays readable. ----
    let alloc_vmem = || -> Result<Value, Box<dyn Error>> {
        let op = OperationBuilder::new("memref.alloc", location)
            .add_results(&[vmem])
            .build()?;
        Ok(block.append_operation(op)?.result(0).unwrap().into())
    };
    let a_v = alloc_vmem()?;
    let b_v = alloc_vmem()?;
    let c_v = alloc_vmem()?;

    // ---- the tiled nest ----
    let ti_body = {
        let ti_block = Block::new(&[(Type::index(context), location)]);
        let ti: Value = ti_block.argument(0)?.into();

        let tj_body = {
            let tj_block = Block::new(&[(Type::index(context), location)]);
            let tj: Value = tj_block.argument(0)?.into();

            // tile.zero %c_v — the vector unit fills the accumulator tile.
            // (C[i,j] += A[i,k]·B[k,j] accumulates across the K loop, so the
            // buffer must start at zero — same as acc in the Triton kernel.)
            ti_block.append_operation(
                OperationBuilder::new("tile.zero", location)
                    .add_operands(&[c_v])
                    .build()?,
            )?;

            let tk_body = {
                let tk_block = Block::new(&[(Type::index(context), location)]);
                let tk: Value = tk_block.argument(0)?.into();

                // "tile.dma" — the load from 0.1, made chunky and explicit.
                // Real Mosaic: tpu_dma_start / tpu_dma_wait with semaphores,
                // double-buffering the K loop (see tpu-double-buffer.svg).
                let dma = |direction: &str,
                           src: Value,
                           si: Value,
                           sj: Value,
                           dst: Value|
                 -> Result<(), Box<dyn Error>> {
                    tk_block.append_operation(
                        OperationBuilder::new("tile.dma", location)
                            .add_operands(&[src, si, sj, dst])
                            .add_attributes(&[(
                                Identifier::new(context, "direction"),
                                StringAttribute::new(context, direction).into(),
                            )])
                            .build()?,
                    )?;
                    Ok(())
                };
                // A tile [ti, tk] and B tile [tk, tj] stream into VMEM.
                dma("hbm_to_vmem", a_hbm, ti, tk, a_v)?;
                dma("hbm_to_vmem", b_hbm, tk, tj, b_v)?;

                // tile.mxm — ONE op for 128×128×128 FMA steps. This is the
                // systolic "instruction"; everything else exists to feed it.
                // Lowering to scalar loops here (like 0.1) would leave the MXU
                // idle — the anti-pattern Part B warns about.
                tk_block.append_operation(
                    OperationBuilder::new("tile.mxm", location)
                        .add_operands(&[c_v, a_v, b_v])
                        .build()?,
                )?;

                tk_block
                    .append_operation(OperationBuilder::new("scf.yield", location).build()?)?;
                let region = Region::new();
                region.append_block(tk_block);
                region
            };

            tj_block.append_operation(
                OperationBuilder::new("scf.for", location)
                    .add_operands(&[c0, c_tiles, c1])
                    .add_regions([tk_body])
                    .build()?,
            )?;

            // ---- epilogue slot ----
            // On TPU this is where the *vector unit* would fuse bias, GELU/SiLU
            // (TinyMoE kernel 4!), or dequant scales — while the tile is still
            // in VMEM, before the write-back DMA. Fusion wins are bandwidth
            // wins (see the roofline in post 1.1).
            //
            //   "tile.epilogue"(%c_v) {op = "silu+bias"}  // illustrative

            // Result tile → HBM.
            tj_block.append_operation(
                OperationBuilder::new("tile.dma", location)
                    .add_operands(&[c_v, ti, tj, c_hbm])
                    .add_attributes(&[(
                        Identifier::new(context, "direction"),
                        StringAttribute::new(context, "vmem_to_hbm").into(),
                    )])
                    .build()?,
            )?;
            tj_block.append_operation(OperationBuilder::new("scf.yield", location).build()?)?;
            let region = Region::new();
            region.append_block(tj_block);
            region
        };

        ti_block.append_operation(
            OperationBuilder::new("scf.for", location)
                .add_operands(&[c0, c_tiles, c1])
                .add_regions([tj_body])
                .build()?,
        )?;
        ti_block.append_operation(OperationBuilder::new("scf.yield", location).build()?)?;
        let region = Region::new();
        region.append_block(ti_block);
        region
    };

    block.append_operation(
        OperationBuilder::new("scf.for", location)
            .add_operands(&[c0, c_tiles, c1])
            .add_regions([ti_body])
            .build()?,
    )?;
    block.append_operation(func::r#return(&[], location))?;

    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "expert_gemm"),
        TypeAttribute::new(FunctionType::new(context, &[hbm, hbm, hbm], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}
