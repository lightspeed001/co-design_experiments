//! tinymoe-lowerer — companion to post **0.1 "One Stack, Many ASICs"**.
//!
//! This binary shows the series' central mechanic: a kernel written once at the
//! *tile level* (`tile.gemm`, our stand-in for Triton's `tt` dialect) is lowered
//! to explicit memory traffic and loops (`scf`/`arith`/`memref`) by **Rust code
//! driving MLIR through Melior** — no C++ required for the rewrite itself.
//!
//! Before / after (abridged; `cargo run` prints the real thing):
//!
//! ```text
//! // ---- before: tile-level IR (the portable middle) ----
//! func.func @expert_gemm(%a: memref<16x16xf32>, %b: memref<16x16xf32>,
//!                        %c: memref<16x16xf32>) {
//!   "tile.gemm"(%a, %b, %c) {m = 16 : index, n = 16 : index, k = 16 : index}
//!     : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
//!   return
//! }
//!
//! // ---- after: lower_tile_gemm (what an ASIC compiler's tiler emits) ----
//! func.func @expert_gemm(%a: ..., %b: ..., %c: ...) {
//!   %c0 = arith.constant 0 : index
//!   %c1 = arith.constant 1 : index
//!   %c16 = arith.constant 16 : index
//!   scf.for %i = %c0 to %c16 step %c1 {
//!     scf.for %j = %c0 to %c16 step %c1 {
//!       scf.for %k = %c0 to %c16 step %c1 {
//!         %0 = memref.load %a[%i, %k] : memref<16x16xf32>
//!         %1 = memref.load %b[%k, %j] : memref<16x16xf32>
//!         %2 = arith.mulf %0, %1 : f32
//!         %3 = memref.load %c[%i, %j] : memref<16x16xf32>
//!         %4 = arith.addf %3, %2 : f32
//!         memref.store %4, %c[%i, %j] : memref<16x16xf32>
//!       }
//!     }
//!   }
//!   return
//! }
//! ```
//!
//! Production note: a real compiler discovers ops to rewrite by *walking the
//! IR* and extracts `m`/`n`/`k` from op attributes or static memref shapes.
//! Melior supports both; to keep this file focused on the *emission* side, the
//! two builders share a `GemmSpec` struct that mirrors the tile op's fields.

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

/// Shapes of the toy GEMM. Small enough that the printed IR is readable;
/// real kernels tile these up into scratchpad-sized blocks (posts 1.2, 2.2...).
const M: i64 = 16;
const N: i64 = 16;
const K: i64 = 16;

/// Mirrors the fields of a `tile.gemm` op. In production these come off the IR
/// (attributes, then verified against the memref shapes); here the front-end
/// "knows" them and both IRs are generated from the same facts.
struct GemmSpec {
    m: i64,
    n: i64,
    k: i64,
}

impl GemmSpec {
    fn toy() -> Self {
        Self { m: M, n: N, k: K }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // ------------------------------------------------------------------
    // Context setup. `register_all_dialects` + `load_all_available_dialects`
    // gives us func/arith/scf/memref; `set_allow_unregistered_dialects` lets
    // our toy `tile.*` ops exist without a C++/TableGen dialect definition.
    //
    // WHY THIS MATTERS: a production portable IR (Triton's `tt`, Tenstorrent's
    // `ttir`, Neuron's tile ops) is a *registered* C++ dialect with verifiers
    // and custom syntax. Unregistered ops are the pure-Rust prototyping lane.
    // ------------------------------------------------------------------
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);

    let spec = GemmSpec::toy();

    // Phase 1 — emit the tile-level module (the portable middle).
    let before = build_tile_level(&context, &spec)?;
    println!("// ---- before: tile-level IR (portable middle) ----");
    println!("{}", before.as_operation());
    assert!(before.as_operation().verify(), "tile-level module must verify");

    // Phase 2 — lower to loops + explicit memory traffic.
    let mut after = build_lowered(&context, &spec)?;

    // Phase 3 — run *upstream MLIR passes* from Rust. This is half the point
    // of parking our IR in MLIR: canonicalization/CSE/etc. are free.
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(transform::create_canonicalizer_pass());
    pass_manager.add_pass(transform::create_cse_pass());
    pass_manager.run(&mut after)?;

    println!("\n// ---- after: lower_tile_gemm + canonicalize + cse ----");
    println!("{}", after.as_operation());
    assert!(after.as_operation().verify(), "lowered module must verify");

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 1: tile-level IR.
//
// One function, one tile op: `tile.gemm %a, %b, %c` with dims as attributes.
// Tile-level IR says WHAT (a16x16 @ 16x16 accumulate) and stays silent about
// loops, buffers, and movement — every ASIC post in the series asks "what does
// this op become on YOUR machine?".
// ---------------------------------------------------------------------------
fn build_tile_level(context: &Context, spec: &GemmSpec) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let memref = Type::parse(context, "memref<16x16xf32>")?;
    let index_type = Type::index(context);

    let module = Module::new(location);

    // Function body: three memref arguments.
    let block = Block::new(&[(memref, location), (memref, location), (memref, location)]);
    let a: Value = block.argument(0)?.into();
    let b: Value = block.argument(1)?.into();
    let c: Value = block.argument(2)?.into();

    // tile.gemm — an unregistered op standing in for Triton's tile dialect.
    // Dims ride along as attributes so the printed IR is self-describing.
    let gemm = OperationBuilder::new("tile.gemm", location)
        .add_operands(&[a, b, c])
        .add_attributes(&[
            (Identifier::new(context, "m"), IntegerAttribute::new(spec.m, index_type).into()),
            (Identifier::new(context, "n"), IntegerAttribute::new(spec.n, index_type).into()),
            (Identifier::new(context, "k"), IntegerAttribute::new(spec.k, index_type).into()),
        ])
        .build()?;
    block.append_operation(gemm)?;
    block.append_operation(func::r#return(&[], location))?;

    let region = Region::new();
    region.append_block(block);

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "expert_gemm"),
        TypeAttribute::new(
            FunctionType::new(context, &[memref, memref, memref], &[]).into(),
        ),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}

// ---------------------------------------------------------------------------
// Phase 2: the lowering itself.
//
// `lower_tile_gemm` expands one tile op into a loop nest with explicit
// memref.load/store. THIS is the decision point every ASIC vendor re-implements:
//   - loop order & tile sizes  → what fits the scratchpad (TPU VMEM, Neuron SBUF)
//   - where the loads live     → who moves data (DMA engines vs the compiler)
//   - what fuses into the loop → epilogues, activations, epilogue fusion
// The scf/arith/memref form here is deliberately generic: it is the *last
// common denominator* before a target dialect takes over.
// ---------------------------------------------------------------------------
fn build_lowered(context: &Context, spec: &GemmSpec) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let f32_type = Type::float32(context);
    let memref = Type::parse(context, "memref<16x16xf32>")?;
    let index_type = Type::index(context);

    let module = Module::new(location);
    let block = Block::new(&[(memref, location), (memref, location), (memref, location)]);
    let a: Value = block.argument(0)?.into();
    let b: Value = block.argument(1)?.into();
    let c: Value = block.argument(2)?.into();

    // ---- Loop bounds as SSA constants (arith.constant N : index) ----
    // scf.for takes (lb, ub, step) as operands, so we materialize constants in
    // the entry block where they dominate the loop nest.
    let push_const = |value: i64| -> Result<Value, Box<dyn Error>> {
        let op = OperationBuilder::new("arith.constant", location)
            .add_attributes(&[(
                Identifier::new(context, "value"),
                IntegerAttribute::new(value, index_type).into(),
            )])
            .add_results(&[index_type])
            .build()?;
        Ok(block.append_operation(op)?.result(0).unwrap().into())
    };
    let c0 = push_const(0)?;
    let c1 = push_const(1)?;
    let cm = push_const(spec.m)?;
    let cn = push_const(spec.n)?;
    let ck = push_const(spec.k)?;

    // ---- scf.for %iv = lb to ub step step { ... } ----
    // Region contract (verifier-enforced): exactly one block, one `index` block
    // argument (the induction variable), terminated by `scf.yield`. Build the
    // body block first, then move it into the region — classic nesting pattern.
    //
    //   scf.for %i ... { scf.for %j ... { scf.for %k ... { C[i,j] += A[i,k]*B[k,j] } } }
    let i_body = {
        let i_block = Block::new(&[(index_type, location)]);
        let i: Value = i_block.argument(0)?.into();

        let j_body = {
            let j_block = Block::new(&[(index_type, location)]);
            let j: Value = j_block.argument(0)?.into();

            let k_body = {
                let k_block = Block::new(&[(index_type, location)]);
                let k: Value = k_block.argument(0)?.into();

                // %0 = memref.load %a[%i, %k]
                let a_ik: Value = k_block
                    .append_operation(
                        OperationBuilder::new("memref.load", location)
                            .add_operands(&[a, i, k])
                            .add_results(&[f32_type])
                            .build()?,
                    )?
                    .result(0)
                    .unwrap()
                    .into();
                // %1 = memref.load %b[%k, %j]
                let b_kj: Value = k_block
                    .append_operation(
                        OperationBuilder::new("memref.load", location)
                            .add_operands(&[b, k, j])
                            .add_results(&[f32_type])
                            .build()?,
                    )?
                    .result(0)
                    .unwrap()
                    .into();
                // %2 = arith.mulf %0, %1
                let prod: Value = k_block
                    .append_operation(
                        OperationBuilder::new("arith.mulf", location)
                            .add_operands(&[a_ik, b_kj])
                            .add_results(&[f32_type])
                            .build()?,
                    )?
                    .result(0)
                    .unwrap()
                    .into();
                // %3 = memref.load %c[%i, %j]
                let c_ij: Value = k_block
                    .append_operation(
                        OperationBuilder::new("memref.load", location)
                            .add_operands(&[c, i, j])
                            .add_results(&[f32_type])
                            .build()?,
                    )?
                    .result(0)
                    .unwrap()
                    .into();
                // %4 = arith.addf %3, %2   (accumulate into C)
                let sum: Value = k_block
                    .append_operation(
                        OperationBuilder::new("arith.addf", location)
                            .add_operands(&[c_ij, prod])
                            .add_results(&[f32_type])
                            .build()?,
                    )?
                    .result(0)
                    .unwrap()
                    .into();
                // memref.store %4, %c[%i, %j]
                k_block.append_operation(
                    OperationBuilder::new("memref.store", location)
                        .add_operands(&[sum, c, i, j])
                        .build()?,
                )?;
                // Every scf.for body ends in scf.yield (no iter_args here).
                k_block
                    .append_operation(OperationBuilder::new("scf.yield", location).build()?)?;

                let region = Region::new();
                region.append_block(k_block);
                region
            };

            j_block.append_operation(
                OperationBuilder::new("scf.for", location)
                    .add_operands(&[c0, ck, c1])
                    .add_regions([k_body])
                    .build()?,
            )?;
            j_block.append_operation(OperationBuilder::new("scf.yield", location).build()?)?;

            let region = Region::new();
            region.append_block(j_block);
            region
        };

        i_block.append_operation(
            OperationBuilder::new("scf.for", location)
                .add_operands(&[c0, cn, c1])
                .add_regions([j_body])
                .build()?,
        )?;
        i_block.append_operation(OperationBuilder::new("scf.yield", location).build()?)?;

        let region = Region::new();
        region.append_block(i_block);
        region
    };

    // The outermost i-loop lands in the function body, then we return.
    block.append_operation(
        OperationBuilder::new("scf.for", location)
            .add_operands(&[c0, cm, c1])
            .add_regions([i_body])
            .build()?,
    )?;
    block.append_operation(func::r#return(&[], location))?;

    let region = Region::new();
    region.append_block(block);

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "expert_gemm"),
        TypeAttribute::new(
            FunctionType::new(context, &[memref, memref, memref], &[]).into(),
        ),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}
