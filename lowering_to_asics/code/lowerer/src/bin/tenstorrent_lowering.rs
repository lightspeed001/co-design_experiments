//! tenstorrent_lowering — companion to post **4.2 "Lowering in the Open"**.
//!
//! Fifth shape of `tile.gemm`, and the narrative twist the series has been
//! building to: **the loops come back.** A Tensix core is a small RISC-V CPU
//! with a matrix engine, a vector unit, and a network router bolted on. Its
//! natural program is *firmware* — so the target form of our GEMM is a
//! **grid of cores, each running a loop**, moving tiles over an on-die
//! network (NoC), with explicit barriers. Compare the four targets:
//!
//! ```text
//! // 0.1   loops over elements (generic)          — programs
//! // 1.2   loops over tiles + one big MXU op      — feeding an array
//! // 2.2   loops over tiles + engine cast + PSUM  — directing a cast
//! // 3.2   NO loops; a slot-stamped timetable     — writing time
//! // 4.2   loops return, on a grid, with NoC + barriers — programming machines
//! ```
//!
//! ```text
//! // ---- after: lower_tile_gemm_to_tensix ----
//! func.func @expert_gemm(%a_dram: memref<256x256xf32, 0 : i32>, …) {
//!   "ttix.grid"() ({
//!   ^bb0(%cx: index, %cy: index):                 // one body per core (x, y)
//!     %a_l1 = memref.alloc() : memref<128x128xf32, 1 : i32>   // core-local L1
//!     %b_l1 = memref.alloc() : memref<128x128xf32, 1 : i32>
//!     %c_l1 = memref.alloc() : memref<128x128xf32, 1 : i32>
//!     "ttix.fill"(%c_l1) {unit = "vector"}
//!     scf.for %k = %c0 to %c2 step %c1 {          // loops = firmware
//!       "ttix.noc_read"(%a_dram, %cx, %k, %a_l1) {unit = "noc"}
//!       "ttix.noc_read"(%b_dram, %k, %cy, %b_l1) {unit = "noc"}
//!       "ttix.mxm"(%c_l1, %a_l1, %b_l1) {unit = "matrix"}
//!     }
//!     "ttix.noc_write"(%c_l1, %cx, %cy, %c_dram) {unit = "noc"}
//!     "ttix.barrier"() {scope = "grid"}
//!     "ttix.yield"
//!   }) {grid_w = 2 : index, grid_h = 2 : index}
//!   return
//! }
//! ```
//!
//! New facts the target forces into the IR (the "leak" list of the post):
//! * **grid coordinates** — where compute happens is part of the program;
//! * **NoC moves** — data movement is a first-class op, not a DMA afterthought;
//! * **barriers** — time is back, but as *your* synchronization;
//! * **kernel ABI** — the body is RISC-V firmware; real tt-mlir also pins
//!   packer/unpacker config and operand layouts (we skip those details).
//!
//! Naming note: `ttix.*` is our stand-in dialect, nodding at Tenstorrent's
//! real (open-source!) `ttir`/`ttnn`/`ttkernel`/`ttmetal` dialects in
//! `tenstorrent/tt-mlir`. Read theirs — it's on GitHub.
//!
//! `cargo run --bin tenstorrent_lowering` prints before/after.

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

/// GEMM shapes: 256³ with 128 tiles → a 2×2 grid of cores, one output tile
/// each, K loop of 2 panels. Real sizes are bigger; the mapping is the same.
const M: i64 = 256;
const N: i64 = 256;
const K: i64 = 256;
const TILE: i64 = 128;

/// Memory spaces: card DRAM (space 0) and core-local L1 (space 1). Unlike
/// Groq's SRAM-only world, the model does NOT have to fit in L1 — tiles do.
/// Unlike TPU/Trainium, the scratchpad is a CPU's local memory: firmware
/// addresses it directly.
const DRAM: &str = "0 : i32";
const L1: &str = "1 : i32";

/// The Tensix "cast" (it persists from Trainium, minus the surprise): the
/// matrix engine runs matmuls, the vector unit does fill/epilogues, the NoC
/// moves data, and RISC-V firmware orchestrates all of it.
const UNIT_MATRIX: &str = "matrix";
const UNIT_VECTOR: &str = "vector";
const UNIT_NOC: &str = "noc";

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);

    // Phase 1 — tile-level input (unchanged for the fifth time; by now this
    // IS the joke and the argument).
    let before = build_tile_level(&context)?;
    println!("// ---- before: tile-level IR (portable middle) ----");
    println!("{}", before.as_operation());
    assert!(before.as_operation().verify(), "tile-level module must verify");

    // Phase 2 — lower to a core grid with NoC moves and barriers.
    let mut after = build_tensix_grid(&context)?;

    // Phase 3 — upstream passes from Rust (unchanged machinery).
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(transform::create_canonicalizer_pass());
    pass_manager.add_pass(transform::create_cse_pass());
    pass_manager.run(&mut after)?;

    println!("\n// ---- after: lower_tile_gemm_to_tensix + passes ----");
    println!("{}", after.as_operation());
    assert!(after.as_operation().verify(), "Tensix-shaped module must verify");

    Ok(())
}

/// memref type with a memory space: `memref<128x128xf32, 1 : i32>` (L1).
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

/// Attach string attributes (`unit`, `scope`, `direction`, …) to a builder.
/// Tenstorrent keeps Trainium's cast idea (`unit`) and adds scheduling nouns
/// (`scope = "grid"` on the barrier).
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

/// Allocate a tile buffer in L1 (`memref.alloc` whose result type carries the
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

/// One NoC read: `ttix.noc_read(%src, %i, %j, %dst)` — fetch tile [i, j] of
/// the source matrix into a local buffer. The operand indices ARE the
/// data-movement plan — on a mesh network, "who reads what from where" is
/// part of the program, not a runtime detail.
fn noc_read(
    context: &Context,
    location: Location,
    block: &Block<'_, '_>,
    src: Value,
    si: Value,
    sj: Value,
    dst: Value,
) -> Result<(), Box<dyn Error>> {
    block.append_operation(
        with_attrs(
            context,
            OperationBuilder::new("ttix.noc_read", location)
                .add_operands(&[src, si, sj, dst]),
            &[("unit", UNIT_NOC), ("direction", "dram_to_l1")],
        )
        .build()?,
    )?;
    Ok(())
}

/// Phase 1: `"tile.gemm"(%a, %b, %c) {m, n, k}` on DRAM-resident memrefs.
/// Byte-for-byte the same shape of code as 0.1 / 1.2 / 2.2 / 3.2.
fn build_tile_level(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let dram = memref(context, M, N, DRAM)?;
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

/// Phase 2: the Tensix-shaped rewrite — a grid of cores, each running
/// firmware. The `ttix.grid` body executes once per core; its block
/// arguments are that core's (x, y) grid coordinates (the stand-in for
/// tt-metal's program_id / noc coordinates).
///
/// Inside the body the story of this post plays out:
///   * L1 buffers are allocated *per core* (memref space 1);
///   * `ttix.noc_read`/`ttix.noc_write` move tiles between DRAM and L1 —
///     data movement is a first-class op, because the NoC is a first-class
///     machine part;
///   * the K loop is a real `scf.for` again — firmware runs loops;
///   * `ttix.mxm` is one matrix-engine command per tile (like `tile.mxm`
///     in 1.2, but issued from RISC-V code, on this core);
///   * `ttix.barrier` synchronizes the grid — *time* returns as explicit
///     sync, not as Groq's baked-in schedule.
fn build_tensix_grid(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let dram = memref(context, M, N, DRAM)?;
    let l1_tile = memref(context, TILE, TILE, L1)?;
    let index_type = Type::index(context);

    let module = Module::new(location);
    let block = Block::new(&[(dram, location), (dram, location), (dram, location)]);
    let a_dram: Value = block.argument(0)?.into();
    let b_dram: Value = block.argument(1)?.into();
    let c_dram: Value = block.argument(2)?.into();

    // ---- the per-core kernel body (one region, executed per core) ----
    let body = {
        let core = Block::new(&[(index_type, location), (index_type, location)]);
        let cx: Value = core.argument(0)?.into(); // this core's grid x
        let cy: Value = core.argument(1)?.into(); // this core's grid y

        // Core-local L1 buffers. On Tensix, L1 is a CPU's local memory —
        // firmware addresses it; the compiler (or you) decides what lives
        // here. 128×128 f32 tiles: 64 KiB each, well inside a Tensix L1.
        let a_l1 = alloc_tile(&core, location, l1_tile)?;
        let b_l1 = alloc_tile(&core, location, l1_tile)?;
        let c_l1 = alloc_tile(&core, location, l1_tile)?;

        // vector-unit fill: zero the accumulator tile
        core.append_operation(
            with_attrs(
                context,
                OperationBuilder::new("ttix.fill", location).add_operands(&[c_l1]),
                &[("unit", UNIT_VECTOR)],
            )
            .build()?,
        )?;

        // ---- the K loop: firmware, not a schedule ----
        let c0 = index_const(context, &core, location, 0)?;
        let c1 = index_const(context, &core, location, 1)?;
        let c_panels = index_const(context, &core, location, K / TILE)?;

        let k_body = {
            let k_block = Block::new(&[(index_type, location)]);
            let k: Value = k_block.argument(0)?.into();

            // NoC reads: this core fetches A tile [cx, k] and B tile [k, cy].
            // The operand indices ARE the data-movement plan (see `noc_read`).
            noc_read(context, location, &k_block, a_dram, cx, k, a_l1)?;
            noc_read(context, location, &k_block, b_dram, k, cy, b_l1)?;

            // Matrix-engine command: 128×128×128 on this core's engine,
            // accumulating into the L1 tile. Same op shape as 1.2's
            // tile.mxm — but issued from firmware, per core.
            k_block.append_operation(
                with_attrs(
                    context,
                    OperationBuilder::new("ttix.mxm", location)
                        .add_operands(&[c_l1, a_l1, b_l1]),
                    &[("unit", UNIT_MATRIX)],
                )
                .build()?,
            )?;

            k_block.append_operation(OperationBuilder::new("scf.yield", location).build()?)?;
            let region = Region::new();
            region.append_block(k_block);
            region
        };

        core.append_operation(
            OperationBuilder::new("scf.for", location)
                .add_operands(&[c0, c_panels, c1])
                .add_regions([k_body])
                .build()?,
        )?;

        // NoC write-back: result tile [cx, cy] → DRAM.
        core.append_operation(
            with_attrs(
                context,
                OperationBuilder::new("ttix.noc_write", location)
                    .add_operands(&[c_l1, cx, cy, c_dram]),
                &[("unit", UNIT_NOC), ("direction", "l1_to_dram")],
            )
            .build()?,
        )?;

        // Grid barrier: everyone's tiles are written before anyone reads.
        // Groq baked this into the timetable; here it's a visible op — the
        // synchronization is *yours*, which is the open-machine deal.
        core.append_operation(
            with_attrs(
                context,
                OperationBuilder::new("ttix.barrier", location),
                &[("scope", "grid")],
            )
            .build()?,
        )?;

        core.append_operation(OperationBuilder::new("ttix.yield", location).build()?)?;
        let region = Region::new();
        region.append_block(core);
        region
    };

    // The grid op: body runs once per (x, y). 2×2 cores, one output tile each.
    block.append_operation(
        OperationBuilder::new("ttix.grid", location)
            .add_regions([body])
            .add_attributes(&[
                (
                    Identifier::new(context, "grid_w"),
                    IntegerAttribute::new(2, index_type).into(),
                ),
                (
                    Identifier::new(context, "grid_h"),
                    IntegerAttribute::new(2, index_type).into(),
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
        TypeAttribute::new(FunctionType::new(context, &[dram, dram, dram], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}
