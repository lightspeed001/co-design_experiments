//! mtia_lowering — companion to post **5.2 "Triton for MTIA"**.
//!
//! Sixth shape of `tile.gemm`, and the series' inversion: until now WE defined
//! both sides of every lowering (our `tile.*` in, our target shape out). The
//! production case study this binary imitates — Meta's *Triton for MTIA* work
//! (arXiv 2608.00325) — has a **fixed, GPU-shaped source language** (Triton)
//! and a **non-GPU target**. The lowering can't redesign the source; it must
//! *negotiate*. The paper's headline: negotiation is possible with **minimal
//! language extensions** — and this binary shows the shape of that claim.
//!
//! So the sixth shape is the **minimal-rewrite** shape. Compare the arc:
//!
//! ```text
//! // 1.2   TPU       — fat tiles (rewrite the innermost op)
//! // 2.2   Trainium  — cast + second memory (rewrite the facts)
//! // 3.2   Groq      — delete the loops (rewrite everything)
//! // 4.2   Tensix    — loops return on a grid (rewrite the topology)
//! // 5.2   MTIA      — keep the tile program; fill THREE gaps
//! ```
//!
//! ```text
//! // ---- after: lower_tile_gemm_to_mtia ----
//! "mtia.kernel"() ({
//! ^bb0(%pid: index):                        // Triton's program_id — kept!
//!   // GAP 1 (compiler math): Triton hands us ONE linear program_id;
//!   // the target's grid is 2D. Fill it with divsi/remsi/muli:
//!   %ti = arith.divsi %pid, %c2 ; %tj = arith.remsi %pid, %c2
//!   %si = arith.muli %ti, %c128 ; %sj = arith.muli %tj, %c128
//!   // GAP 2 (explicit residency): on-chip SRAM is not implicit.
//!   %a_s = memref.alloc() : memref<128x128xf32, 1 : i32>
//!   ...
//!   scf.for %k = %c0 to %c2 step %c1 {
//!     "mtia.load_tile"(%a, %si, %sk, %a_s)   {unit = "dma"}
//!     "mtia.load_tile"(%b, %sk, %sj, %b_s)   {unit = "dma"}
//!     "mtia.tensor_op"(%c_s, %a_s, %b_s)     {unit = "tensor", shape = "128x128x128"}
//!     "mtia.ext_sync"() {scope = "tile"}     // GAP 3: language extension
//!   }
//!   "mtia.store_tile"(%c_s, %si, %sj, %c)    {unit = "dma"}
//!   "mtia.yield"
//! }) {grid = 4 : index, mapping = "linear_to_2d"}
//! ```
//!
//! The three gaps, named as the post names them:
//!   1. **grid mapping** — filled by *compiler math* (the arith ops above);
//!   2. **SRAM residency** — filled by *types* (memory space 1, alloc ops);
//!   3. **features Triton can't say** — filled by *language extensions*
//!      (`mtia.ext_*`). Here `ext_sync`; the MoE combine wants
//!      `ext_grid_reduce` (see the post's illustrative snippet).
//!
//! Naming note: `mtia.*` is our stand-in dialect. The real MTIA backend is
//! C++ MLIR inside PyTorch/TorchInductor (read the paper — it's open access).
//!
//! `cargo run --bin mtia_lowering` prints before/after.

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

/// GEMM shapes: 256³, tiles of 128 → grid of 4 kernels (2×2). Triton's grid
/// is *linear* (`program_id(0)`), so the mapping gap is real and visible.
const M: i64 = 256;
const N: i64 = 256;
const K: i64 = 256;
const TILE: i64 = 128;
const GRID: i64 = (M / TILE) * (N / TILE); // 4 kernels in the launch grid

/// Memory spaces. MTIA bets on *cheap capacity* (LPDDR-class off-chip) behind
/// a generous on-chip SRAM — different memory philosophy from Maia's HBM
/// (post 5.1 compares the two bets).
const LPDDR: &str = "0 : i32"; // off-chip
const SRAM: &str = "1 : i32"; // on-chip (explicitly allocated — gap 2)

/// The cast, inherited from Trainium/Tenstorrent and unremarkable by now:
/// dma moves, tensor computes, vector fills.
const UNIT_DMA: &str = "dma";
const UNIT_TENSOR: &str = "tensor";
const UNIT_VECTOR: &str = "vector";

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);

    // Phase 1 — tile-level input (sixth time's the charm).
    let before = build_tile_level(&context)?;
    println!("// ---- before: tile-level IR (portable middle) ----");
    println!("{}", before.as_operation());
    assert!(before.as_operation().verify(), "tile-level module must verify");

    // Phase 2 — the minimal-rewrite lowering with gap fillers.
    let mut after = build_mtia_kernel(&context)?;

    // Phase 3 — upstream passes from Rust (unchanged).
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(transform::create_canonicalizer_pass());
    pass_manager.add_pass(transform::create_cse_pass());
    pass_manager.run(&mut after)?;

    println!("\n// ---- after: lower_tile_gemm_to_mtia + passes ----");
    println!("{}", after.as_operation());
    assert!(after.as_operation().verify(), "MTIA-shaped module must verify");

    Ok(())
}

/// memref type with a memory space: `memref<128x128xf32, 1 : i32>` (SRAM).
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

/// Push a binary `index` op (`arith.divsi` / `arith.remsi` / `arith.muli`)
/// and return its result. These four lines of "compiler math" are gap filler
/// #1: the GPU-shaped source gives us one linear `program_id`; the target's
/// grid is 2D, so the lowering *computes* the coordinates the source never
/// provided. (Tenstorrent got them as block arguments; MTIA gets arithmetic.)
fn index_binop(
    context: &Context,
    block: &Block<'_, '_>,
    location: Location,
    name: &str,
    lhs: Value,
    rhs: Value,
) -> Result<Value, Box<dyn Error>> {
    let op = OperationBuilder::new(name, location)
        .add_operands(&[lhs, rhs])
        .add_results(&[Type::index(context)])
        .build()?;
    Ok(block.append_operation(op)?.result(0).unwrap().into())
}

/// Attach string attributes (`unit`, `scope`, `shape`, …) to a builder.
fn with_attrs<'c>(
    context: &'c Context,
    builder: OperationBuilder<'c>,
    attrs: &[(&str, &str)],
) -> OperationBuilder<'c> {
    let mut builder = builder;
    for (key, value) in attrs {
        builder = builder.add_attributes(&[(
            Identifier::new(context, key),
            StringAttribute::new(context, value).into(),
        )]);
    }
    builder
}

/// Allocate a tile buffer (`memref.alloc` whose result type carries the
/// memory space) and return its SSA value.
fn alloc_tile(
    block: &Block<'_, '_>,
    location: Location,
    result_type: Type,
) -> Result<Value, Box<dyn Error>> {
    let op = OperationBuilder::new("memref.alloc", location)
        .add_results(&[result_type])
        .build()?;
    Ok(block.append_operation(op)?.result(0).unwrap().into())
}

/// Emit one side-effecting op into a block with a `unit` cast attribute and
/// optional extras. (Same helper shape as groq_lowering's `emit`, minus the
/// clock — MTIA doesn't hand us time, it hands us gaps.)
fn emit(
    context: &Context,
    location: Location,
    block: &Block<'_, '_>,
    unit: &str,
    name: &str,
    operands: &[Value],
    extra: &[(&str, &str)],
) -> Result<(), Box<dyn Error>> {
    let builder = OperationBuilder::new(name, location).add_operands(operands);
    let builder = with_attrs(context, builder, extra);
    let builder = with_attrs(context, builder, &[("unit", unit)]);
    block.append_operation(builder.build()?)?;
    Ok(())
}

/// Phase 1: `"tile.gemm"(%a, %b, %c) {m, n, k}` — the portable middle,
/// unchanged for the sixth and final time in the series' main arc.
fn build_tile_level(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let lpddr = memref(context, M, N, LPDDR)?;
    let index_type = Type::index(context);

    let module = Module::new(location);
    let block = Block::new(&[(lpddr, location), (lpddr, location), (lpddr, location)]);
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
        TypeAttribute::new(FunctionType::new(context, &[lpddr, lpddr, lpddr], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}

/// Phase 2: the minimal-rewrite lowering. The body preserves the *Triton
/// shape* of the program (one `program_id`, a tile loop, load/compute/store)
/// and fills exactly three gaps — mapping math, explicit residency, one
/// extension op. Everything else is untouched on purpose.
fn build_mtia_kernel(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let lpddr = memref(context, M, N, LPDDR)?;
    let sram_tile = memref(context, TILE, TILE, SRAM)?;
    let index_type = Type::index(context);

    let module = Module::new(location);
    let block = Block::new(&[(lpddr, location), (lpddr, location), (lpddr, location)]);
    let a_lpddr: Value = block.argument(0)?.into();
    let b_lpddr: Value = block.argument(1)?.into();
    let c_lpddr: Value = block.argument(2)?.into();

    // ---- the per-kernel body (Triton's program keeps its shape) ----
    let body = {
        let kernel = Block::new(&[(index_type, location)]);
        let pid: Value = kernel.argument(0)?.into(); // Triton's program_id

        // constants for the mapping math and the K loop
        let c0 = index_const(context, &kernel, location, 0)?;
        let c1 = index_const(context, &kernel, location, 1)?;
        let c2 = index_const(context, &kernel, location, 2)?;
        let c128 = index_const(context, &kernel, location, TILE)?;

        // ---- GAP 1: grid mapping, filled by compiler math ----
        // Triton hands us ONE linear program_id; the target's compute grid is
        // 2D. So the lowering computes the coordinates the source never had:
        //   %ti = pid / 2 ; %tj = pid % 2 ; %si = %ti * 128 ; %sj = %tj * 128
        // (Compare 4.2: Tensix cores natively spoke (cx, cy). Here the
        // GPU-shaped source speaks one dimension, and arithmetic bridges it.)
        let ti = index_binop(context, &kernel, location, "arith.divsi", pid, c2)?;
        let tj = index_binop(context, &kernel, location, "arith.remsi", pid, c2)?;
        let si = index_binop(context, &kernel, location, "arith.muli", ti, c128)?;
        let sj = index_binop(context, &kernel, location, "arith.muli", tj, c128)?;

        // ---- GAP 2: explicit SRAM residency ----
        // MTIA's on-chip SRAM is generous but *explicit*: tiles live where the
        // kernel allocates them (memory space 1), fetched from LPDDR by dma.
        let a_s = alloc_tile(&kernel, location, sram_tile)?;
        let b_s = alloc_tile(&kernel, location, sram_tile)?;
        let c_s = alloc_tile(&kernel, location, sram_tile)?;

        // vector-unit fill: zero the accumulator tile
        emit(context, location, &kernel, UNIT_VECTOR, "mtia.fill", &[c_s], &[])?;

        // ---- the tile loop (Triton programs have loops; leave them be) ----
        // K/TILE panels along K; reuse the existing `%c2` (same value) so the
        // dump is CSE-stable either way.
        assert_eq!(K / TILE, 2);
        let c_panels = c2;
        let k_body = {
            let k_block = Block::new(&[(index_type, location)]);
            let k: Value = k_block.argument(0)?.into();

            // sk = k * 128: more mapping math (panel index → byte-ish index)
            let sk = index_binop(context, &k_block, location, "arith.muli", k, c128)?;

            // dma tile fetches — explicit, as every target since 1.2
            emit(
                context,
                location,
                &k_block,
                UNIT_DMA,
                "mtia.load_tile",
                &[a_lpddr, si, sk, a_s],
                &[("direction", "lpddr_to_sram")],
            )?;
            emit(
                context,
                location,
                &k_block,
                UNIT_DMA,
                "mtia.load_tile",
                &[b_lpddr, sk, sj, b_s],
                &[("direction", "lpddr_to_sram")],
            )?;

            // the fixed tensor unit — one op per tile, like tile.mxm (1.2)
            emit(
                context,
                location,
                &k_block,
                UNIT_TENSOR,
                "mtia.tensor_op",
                &[c_s, a_s, b_s],
                &[("shape", "128x128x128")],
            )?;

            // ---- GAP 3: a language extension ----
            // Triton has no word for the target's tile-level sync; the paper's
            // answer is a *minimal extension* rather than an emulation. Our
            // stand-in: mtia.ext_sync. (The MoE combine would want
            // mtia.ext_grid_reduce — see the post.)
            emit(
                context,
                location,
                &k_block,
                UNIT_VECTOR,
                "mtia.ext_sync",
                &[],
                &[("scope", "tile")],
            )?;

            k_block.append_operation(OperationBuilder::new("scf.yield", location).build()?)?;
            let region = Region::new();
            region.append_block(k_block);
            region
        };

        kernel.append_operation(
            OperationBuilder::new("scf.for", location)
                .add_operands(&[c0, c_panels, c1])
                .add_regions([k_body])
                .build()?,
        )?;

        // write the finished tile back
        emit(
            context,
            location,
            &kernel,
            UNIT_DMA,
            "mtia.store_tile",
            &[c_s, si, sj, c_lpddr],
            &[("direction", "sram_to_lpddr")],
        )?;

        kernel.append_operation(OperationBuilder::new("mtia.yield", location).build()?)?;
        let region = Region::new();
        region.append_block(kernel);
        region
    };

    // The kernel-launch op: grid of 4, linear pid (the source's contract).
    block.append_operation(
        OperationBuilder::new("mtia.kernel", location)
            .add_regions([body])
            .add_attributes(&[
                (
                    Identifier::new(context, "grid"),
                    IntegerAttribute::new(GRID, index_type).into(),
                ),
                (
                    Identifier::new(context, "mapping"),
                    StringAttribute::new(context, "linear_to_2d").into(),
                ),
            ])
            .build()?,
    )?;
    block.append_operation(func::r#return(&[], location))?;

    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "expert_gemm"),
        TypeAttribute::new(FunctionType::new(context, &[lpddr, lpddr, lpddr], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}
