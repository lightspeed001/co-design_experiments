//! cerebras_lowering — companion to post **6.2 "Lowering to the Wafer"**.
//!
//! Seventh and final `tile.gemm` lowering of the series' main arc, and the
//! shape where the loop disappears for the second and last time. Groq (3.2)
//! deleted the loop and wrote the machine a *timetable* in time; the wafer
//! deletes it and writes a *placement* in space. A WSE-style mesh has no
//! caches, no runtime scheduler, and no HBM: 900,000 small cores with local
//! SRAM, a nearest-neighbor fabric (Swarm), and weights streamed layer-by-layer
//! from an off-wafer MemoryX appliance. Every op needs a home; the dataflow
//! triggers fire when data arrives. The wafer waits for no one — so the
//! compiler must discover the parallelism and pin it down before the first
//! byte moves.
//!
//! ```text
//! // ---- after: lower_tile_gemm_to_wafer ----
//! "wafer.pipeline"() ({
//!   // weights are streamed, not resident — memory spaces vanish on-wafer
//!   %w = "csl.weight_stream"(%b) {source = "memoryx", layout = "panel_major"}
//!        : (memref<256x256xf32, 0 : i32>) -> memref<256x256xf32>
//!   // the K-loop is gone: it became the streaming pipeline (pipeline = 2)
//!   %acc = "csl.mac_tile"(%a, %w) {place = "pe[0,0]", pipeline = 2 : index,
//!                                  shape = "128x128x128", unit = "mac"}
//!        : (...) -> memref<128x128xf32>          // acc lives ON the core
//!   "csl.stream_out"(%acc, %c) {place = "pe[0,0]", route = "south", unit = "stream"}
//!   // … pe[0,1], pe[1,0], pe[1,1] identical — emitted by the placement loop
//!   "csl.yield"
//! }) {mesh = "2x2", pipeline = 2 : index, routing = "swarm"}
//! ```
//!
//! Three design points, named as the post names them:
//!   1. **the loop became the pipeline** — K/TILE panels stream through the
//!      fabric in flight; `pipeline = 2 : index` is the depth, not a trip count;
//!   2. **residency became movement** — weights arrive from MemoryX (`0 : i32`
//!      stays on the function signature; on-wafer values have NO memory space);
//!   3. **placement is total** — every op carries `place = "pe[r,c]"`. The
//!      Rust loop that emits the four PEs *is* CSL's host-side mesh loop.
//!
//! Naming note: `wafer.pipeline` / `csl.*` are our stand-ins for the real
//! stack (cstorches graph compiler + CSL, Cerebras' Software Language).
//! Sparsity (SparseX: structured 2:4 and fine-grained) is discussed in the
//! post; the one illustrative `csl.sparse_gather` snippet lives there.
//!
//! `cargo run --bin cerebras_lowering` prints before/after.

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

/// GEMM shapes: 256³, tiles of 128 → a 2×2 mesh of PEs, each PE owning one
/// output tile, with K/TILE = 2 panels in flight as the pipeline depth.
const M: i64 = 256;
const N: i64 = 256;
const K: i64 = 256;
const TILE: i64 = 128;
const PE_ROWS: usize = 2;
const PE_COLS: usize = 2;
/// The K-loop, spatialized: this is a *pipeline depth*, not a trip count.
/// (Groq's time-domain analogue was `schedule_span`; this one is in space.)
const PIPELINE: i64 = K / TILE;

/// The wafer's cast, third and final renaming of the series' three roles
/// (move / compute / coordinate): dma→stream (weights & activations travel),
/// tensor→mac (the per-core multiplier), vector→fabric (routing & reduction).
const UNIT_STREAM: &str = "stream";
const UNIT_MAC: &str = "mac";
const UNIT_FABRIC: &str = "fabric";

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);

    // Phase 1 — tile-level input (seventh and last time).
    let before = build_tile_level(&context)?;
    println!("// ---- before: tile-level IR (portable middle) ----");
    println!("{}", before.as_operation());
    assert!(before.as_operation().verify(), "tile-level module must verify");

    // Phase 2 — the spatial-placement lowering.
    let mut after = build_wafer_pipeline(&context)?;

    // Phase 3 — upstream passes from Rust (unchanged).
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(transform::create_canonicalizer_pass());
    pass_manager.add_pass(transform::create_cse_pass());
    pass_manager.run(&mut after)?;

    println!("\n// ---- after: lower_tile_gemm_to_wafer + passes ----");
    println!("{}", after.as_operation());
    assert!(after.as_operation().verify(), "WSE-shaped module must verify");

    Ok(())
}

/// memref with a memory space: `memref<256x256xf32, 0 : i32>` (DRAM side).
fn memref(context: &Context, rows: i64, cols: i64, space: &str) -> Result<Type, Box<dyn Error>> {
    Ok(Type::parse(
        context,
        &format!("memref<{rows}x{cols}xf32, {space}>"),
    )?)
}

/// memref with NO memory space: on-wafer values. Where a value lives is
/// *which core is holding it* — the `place` attribute, not a space number.
/// (Six targets taught us memory spaces; the wafer deletes them.)
fn memref_on_wafer(context: &Context, rows: i64, cols: i64) -> Result<Type, Box<dyn Error>> {
    Ok(Type::parse(context, &format!("memref<{rows}x{cols}xf32>"))?)
}

/// String attribute pair (attr key → attr value).
fn str_attr<'c>(context: &'c Context, key: &str, value: &str) -> (Identifier<'c>, Attribute<'c>) {
    (
        Identifier::new(context, key),
        StringAttribute::new(context, value).into(),
    )
}

/// Integer attribute pair printing as `N : index`.
fn idx_attr<'c>(context: &'c Context, key: &str, value: i64) -> (Identifier<'c>, Attribute<'c>) {
    (
        Identifier::new(context, key),
        IntegerAttribute::new(value, Type::index(context)).into(),
    )
}

/// Phase 1: `"tile.gemm"(%a, %b, %c) {m, n, k}` — the portable middle,
/// unchanged for the seventh and final time of the main arc.
fn build_tile_level(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let dram = memref(context, M, N, "0 : i32")?;
    let index_type = Type::index(context);

    let module = Module::new(location);
    let block = Block::new(&[(dram, location), (dram, location), (dram, location)]);
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
        TypeAttribute::new(FunctionType::new(context, &[dram, dram, dram], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}

/// Phase 2: the spatial-placement lowering. The region describes the *whole
/// pipeline*: weights stream in once, four PEs each mac their output tile as
/// panels arrive (the K-loop lives in the dataflow triggers), and each PE
/// streams its result out. No `scf.for`, no `arith.constant` — the mapping
/// problem that cost MTIA four ops costs the wafer nothing, because placement
/// is explicit. This Rust function's nested loop is CSL's host-side mesh loop.
fn build_wafer_pipeline(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let dram = memref(context, M, N, "0 : i32")?;
    let matrix_on_wafer = memref_on_wafer(context, M, N)?;
    let acc_on_wafer = memref_on_wafer(context, TILE, TILE)?;

    let module = Module::new(location);
    let block = Block::new(&[(dram, location), (dram, location), (dram, location)]);
    let a_dram: Value = block.argument(0)?.into();
    let b_dram: Value = block.argument(1)?.into();
    let c_dram: Value = block.argument(2)?.into();

    // ---- the wafer program (one region: the pipeline is the loop) ----
    let body = {
        let pipeline = Block::new(&[]);

        // 1. weights stream from MemoryX — they never "live" on the wafer.
        //    `panel_major` = the fabric feeds K-panels one after another,
        //    which is what turns the K-loop into pipeline depth.
        let w: Value = pipeline
            .append_operation(
                OperationBuilder::new("csl.weight_stream", location)
                    .add_operands(&[b_dram])
                    .add_results(&[matrix_on_wafer])
                    .add_attributes(&[
                        str_attr(context, "layout", "panel_major"),
                        str_attr(context, "source", "memoryx"),
                        str_attr(context, "unit", UNIT_STREAM),
                    ])
                    .build()?,
            )?
            .result(0)
            .unwrap()
            .into();

        // 2. THE PLACEMENT LOOP (this is CSL's host-side mesh loop, in Rust).
        //    Every (row, col) PE owns one 128×128 output tile of the 256×256
        //    result; each accumulates across the streamed K-panels and ships
        //    its tile south. A bad place = idle fabric, which is why placement
        //    is a first-class attribute and not a runtime decision.
        for row in 0..PE_ROWS {
            for col in 0..PE_COLS {
                let place = format!("pe[{row},{col}]");

                // the per-core multiplier: acc lives on the core (unspaced type)
                let acc: Value = pipeline
                    .append_operation(
                        OperationBuilder::new("csl.mac_tile", location)
                            .add_operands(&[a_dram, w])
                            .add_results(&[acc_on_wafer])
                            .add_attributes(&[
                                str_attr(context, "place", &place),
                                idx_attr(context, "pipeline", PIPELINE),
                                str_attr(context, "shape", "128x128x128"),
                                str_attr(context, "unit", UNIT_MAC),
                            ])
                            .build()?,
                    )?
                    .result(0)
                    .unwrap()
                    .into();

                // stream the finished tile to its home in %c
                pipeline.append_operation(
                    OperationBuilder::new("csl.stream_out", location)
                        .add_operands(&[acc, c_dram])
                        .add_attributes(&[
                            str_attr(context, "place", &place),
                            str_attr(context, "route", "south"),
                            str_attr(context, "unit", UNIT_STREAM),
                        ])
                        .build()?,
                )?;
            }
        }

        // the region's terminator (unregistered regions require one)
        pipeline.append_operation(OperationBuilder::new("csl.yield", location).build()?)?;
        let region = Region::new();
        region.append_block(pipeline);
        region
    };

    // The pipeline-launch op: mesh shape, pipeline depth, fabric routing.
    block.append_operation(
        OperationBuilder::new("wafer.pipeline", location)
            .add_regions([body])
            .add_attributes(&[
                str_attr(context, "mesh", "2x2"),
                idx_attr(context, "pipeline", PIPELINE),
                str_attr(context, "routing", "swarm"),
            ])
            .build()?,
    )?;
    block.append_operation(func::r#return(&[], location))?;

    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "expert_gemm"),
        TypeAttribute::new(FunctionType::new(context, &[dram, dram, dram], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}
