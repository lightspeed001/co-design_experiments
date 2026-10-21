//! trainium_lowering — companion to post **2.2 "Lowering to Trainium"**.
//!
//! Same tile-level input as 0.1/1.2 (`tile.gemm`), third target shape. Two
//! things change relative to the TPU-shaped rewrite (1.2), and both are
//! Trainium nouns that leak into the IR:
//!
//! 1. **A second scratchpad.** Accumulators live in **PSUM** (memory space 2),
//!    not in the SBUF tile (space 1) where the inputs live. `neuron.matmul`
//!    writes partial sums to PSUM; an epilogue moves/finalizes into SBUF.
//! 2. **An explicit cast.** Every op carries `{engine = ...}` — "tensor",
//!    "vector", "dma". The portable IR says *what* to compute; Trainium's IR
//!    says *who computes it*. That cast list is part of the program.
//!
//! ```text
//! // ---- before: tile-level IR (unchanged contract) ----
//! "tile.gemm"(%a, %b, %c) {m = 256, n = 256, k = 256} : (…) -> ()
//!
//! // ---- after: lower_tile_gemm_to_trainium ----
//! scf.for %ti = %c0 to %c2 step %c1 {
//!   scf.for %tj = %c0 to %c2 step %c1 {
//!     "neuron.fill"(%c_psum) {engine = "vector"}         // zero the PSUM acc
//!     scf.for %tk = %c0 to %c2 step %c1 {
//!       "neuron.dma"(%a_hbm, %ti, %tk, %a_sbuf) {engine = "dma", …}
//!       "neuron.dma"(%b_hbm, %tk, %tj, %b_sbuf) {engine = "dma", …}
//!       "neuron.matmul"(%c_psum, %a_sbuf, %b_sbuf) {engine = "tensor"}
//!     }
//!     "neuron.epilogue"(%c_sbuf, %c_psum) {engine = "vector", op = "silu"}
//!     "neuron.dma"(%c_sbuf, %ti, %tj, %c_hbm) {engine = "dma", …}
//!   }
//! }
//! ```
//!
//! Production notes (post 2.2 has the full list):
//! * Real Neuron scheduling is a dataflow of engine queues; `neuron.dma` here
//!   stands in for DMA-engine descriptor chains with SBUF residency checks.
//! * PSUM is fp32-class while SBUF tiles are bf16-class on real silicon; we
//!   keep f32 everywhere so the dump stays readable.
//! * The *cast* is chosen by the compiler in `neuronx-cc`; NKI lets you pin it
//!   by hand. Our `engine` attributes are the hand-pinned version.
//!
//! `cargo run --bin trainium_lowering` prints before/after and runs passes.

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

/// GEMM shapes in Tensor-Engine-friendly multiples (check NKI guidance per
/// generation; 128-tiles are the safe teaching size).
const M: i64 = 256;
const N: i64 = 256;
const K: i64 = 256;
const TILE: i64 = 128;

/// Memory spaces. SBUF and PSUM are Trainium's two on-chip memories: SBUF is
/// the staging scratchpad (like TPU VMEM), PSUM is a *separate* accumulator
/// buffer for matmul partials. The second space is the story of this post.
const HBM: &str = "0 : i32"; // off-chip HBM
const SBUF: &str = "1 : i32"; // scratchpad buffer — tiles of A, B, and results
const PSUM: &str = "2 : i32"; // partial-sum buffer — matmul accumulation target

/// Engine names, as they appear in the cast. "dma" moves data, "tensor" runs
/// matmuls, "vector" does elementwise/reduction work (and epilogues). A fourth
/// engine in real NeuronCore-v2, GPSIMD, handles genuinely custom SIMD — our
/// IR has no slot for it (see the "leak" list in the post).
const ENGINE_DMA: &str = "dma";
const ENGINE_TENSOR: &str = "tensor";
const ENGINE_VECTOR: &str = "vector";

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    // tile.{gemm} in; neuron.{fill,dma,matmul,epilogue} out — all unregistered
    // stand-ins (see post 0.1 for why the prototyping lane is unregistered).
    context.set_allow_unregistered_dialects(true);

    // Phase 1 — tile-level input (same contract as 0.1 and 1.2).
    let before = build_tile_level(&context)?;
    println!("// ---- before: tile-level IR (portable middle) ----");
    println!("{}", before.as_operation());
    assert!(before.as_operation().verify(), "tile-level module must verify");

    // Phase 2 — the Trainium-shaped lowering (engine cast + SBUF/PSUM).
    let mut after = build_trainium_shaped(&context)?;

    // Phase 3 — upstream passes from Rust (unchanged machinery).
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(transform::create_canonicalizer_pass());
    pass_manager.add_pass(transform::create_cse_pass());
    pass_manager.run(&mut after)?;

    println!("\n// ---- after: lower_tile_gemm_to_trainium + passes ----");
    println!("{}", after.as_operation());
    assert!(after.as_operation().verify(), "Trainium-shaped module must verify");

    Ok(())
}

/// memref type with an explicit memory space, e.g. `memref<128x128xf32, 2 : i32>`
/// (PSUM). Buffers that meet on one engine op must agree about where they live.
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
/// Identical shape of code to 1.2 — the portable middle does not change when
/// the target does. That is the series thesis, restated every post.
fn build_tile_level(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let hbm = memref(context, M, N, HBM)?;
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

/// Attach `{engine = "..."}` to an op under construction.
/// The cast list: on Trainium this attribute is not decoration — a matmul
/// without an engine is an unfinished program, like a value without a type.
fn with_engine<'c>(
    context: &'c Context,
    builder: OperationBuilder<'c>,
    engine: &str,
) -> OperationBuilder<'c> {
    builder.add_attributes(&[(
        Identifier::new(context, "engine"),
        StringAttribute::new(context, engine).into(),
    )])
}

/// Phase 2: the Trainium-shaped rewrite.
///
/// Differences from the TPU-shaped version (1.2), all in the same places:
///   * inputs tile into **SBUF** (space 1), accumulation happens in **PSUM**
///     (space 2) — a memory the TPU-shaped IR never had to name;
///   * every compute/move op is **cast** to an engine;
///   * the K loop's `tile.mxm` becomes `neuron.matmul` with an explicit
///     PSUM destination, and the epilogue (`neuron.epilogue`, vector engine)
///     finalizes PSUM → SBUF — where TinyMoE kernel 4 (SwiGLU) would fuse.
fn build_trainium_shaped(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let hbm = memref(context, M, N, HBM)?;
    let sbuf = memref(context, TILE, TILE, SBUF)?;
    let psum = memref(context, TILE, TILE, PSUM)?;

    let module = Module::new(location);
    let block = Block::new(&[(hbm, location), (hbm, location), (hbm, location)]);
    let a_hbm: Value = block.argument(0)?.into();
    let b_hbm: Value = block.argument(1)?.into();
    let c_hbm: Value = block.argument(2)?.into();

    // Loop bounds: 2 tiles per dimension (256 / 128).
    let c0 = index_const(context, &block, location, 0)?;
    let c1 = index_const(context, &block, location, 1)?;
    let c_tiles = index_const(context, &block, location, M / TILE)?;

    // ---- On-chip buffers ----
    // SBUF tiles for A and B (and later the result); PSUM holds the fp32-class
    // accumulator that the Tensor Engine updates every K panel.
    let alloc = |result_type: Type| -> Result<Value, Box<dyn Error>> {
        let op = OperationBuilder::new("memref.alloc", location)
            .add_results(&[result_type])
            .build()?;
        Ok(block.append_operation(op)?.result(0).unwrap().into())
    };
    let a_sbuf = alloc(sbuf)?;
    let b_sbuf = alloc(sbuf)?;
    let c_sbuf = alloc(sbuf)?;
    let c_psum = alloc(psum)?;

    // ---- the tiled nest, with a cast list ----
    let ti_body = {
        let ti_block = Block::new(&[(Type::index(context), location)]);
        let ti: Value = ti_block.argument(0)?.into();

        let tj_body = {
            let tj_block = Block::new(&[(Type::index(context), location)]);
            let tj: Value = tj_block.argument(0)?.into();

            // "neuron.fill" — zero the PSUM accumulator (vector engine owns
            // fill/reduction work). Same reason `acc = tl.zeros(...)` exists in
            // the Triton kernel: C += A@B accumulates across the K loop.
            tj_block.append_operation(
                with_engine(
                    context,
                    OperationBuilder::new("neuron.fill", location)
                        .add_operands(&[c_psum]),
                    ENGINE_VECTOR,
                )
                .build()?,
            )?;

            let tk_body = {
                let tk_block = Block::new(&[(Type::index(context), location)]);
                let tk: Value = tk_block.argument(0)?.into();

                // DMA engine stages one tile of A and B per K panel. In real
                // Neuron these are descriptor chains that overlap with compute
                // (the Trainium analogue of 1.2's double-buffering diagram).
                let dma = |direction: &str,
                           src: Value,
                           si: Value,
                           sj: Value,
                           dst: Value|
                 -> Result<(), Box<dyn Error>> {
                    tk_block.append_operation(
                        with_engine(
                            context,
                            OperationBuilder::new("neuron.dma", location)
                                .add_operands(&[src, si, sj, dst])
                                .add_attributes(&[(
                                    Identifier::new(context, "direction"),
                                    StringAttribute::new(context, direction).into(),
                                )]),
                            ENGINE_DMA,
                        )
                        .build()?,
                    )?;
                    Ok(())
                };
                dma("hbm_to_sbuf", a_hbm, ti, tk, a_sbuf)?;
                dma("hbm_to_sbuf", b_hbm, tk, tj, b_sbuf)?;

                // "neuron.matmul" — Tensor Engine. Note the operand order the
                // target wants: (accumulator, lhs, rhs) with the accumulator in
                // PSUM. The portable `tile.gemm` never mentioned PSUM; this op
                // cannot exist without it.
                tk_block.append_operation(
                    with_engine(
                        context,
                        OperationBuilder::new("neuron.matmul", location)
                            .add_operands(&[c_psum, a_sbuf, b_sbuf]),
                        ENGINE_TENSOR,
                    )
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

            // "neuron.epilogue" — vector engine finalizes PSUM → SBUF. The
            // `{op = "silu"}` attribute is TinyMoE kernel 4 fusing right here:
            // elementwise work rides the accumulator path instead of round-
            // tripping through HBM. (Real NKI: `nl.add`/activation on the
            // vector engine between `nl.matmul` and `nl.store`.)
            tj_block.append_operation(
                with_engine(
                    context,
                    OperationBuilder::new("neuron.epilogue", location)
                        .add_operands(&[c_sbuf, c_psum])
                        .add_attributes(&[(
                            Identifier::new(context, "op"),
                            StringAttribute::new(context, "silu").into(),
                        )]),
                    ENGINE_VECTOR,
                )
                .build()?,
            )?;

            // Result tile → HBM (DMA engine).
            tj_block.append_operation(
                with_engine(
                    context,
                    OperationBuilder::new("neuron.dma", location)
                        .add_operands(&[c_sbuf, ti, tj, c_hbm])
                        .add_attributes(&[(
                            Identifier::new(context, "direction"),
                            StringAttribute::new(context, "sbuf_to_hbm").into(),
                        )]),
                    ENGINE_DMA,
                )
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
