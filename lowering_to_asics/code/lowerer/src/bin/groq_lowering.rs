//! groq_lowering — companion to post **3.2 "Lowering to the LPU"**.
//!
//! Same tile-level input as ever (`tile.gemm`), and then the punchline of the
//! series so far: **the loop nest is gone.** On a Groq-style TSP there is no
//! runtime to speak of — no caches, no scheduler, no dynamic anything. Every
//! op executes in a *time slot* on a *named functional unit*, decided at
//! compile time. So the target form of a GEMM is not loops; it is a **flat,
//! cycle-stamped timetable**.
//!
//! ```text
//! // ---- before: tile-level IR (unchanged contract) ----
//! "tile.gemm"(%a, %b, %c) {m = 256, n = 256, k = 256} : (…) -> ()
//!
//! // ---- after: lower_tile_gemm_to_lpu — a schedule, not a program ----
//! "groq.move"(%a_sram, %c0, %c0, %a_tile) {slot = 0 : index, unit = "mover"}
//! "groq.move"(%b_sram, %c0, %c0, %b_tile) {slot = 1 : index, unit = "mover"}
//! "groq.fill"(%c_tile)                    {slot = 2 : index, unit = "vector"}
//! "groq.matmul"(%c_tile, %a_tile, %b_tile){slot = 3 : index, unit = "matmul"}
//! … one stamped op per (tile, panel), K loop fully unrolled …
//! "groq.epilogue"(%c_out, %c_tile)        {slot = N : index, unit = "vector"}
//! // total slots = N
//! ```
//!
//! Three things to notice, each a post-sized idea:
//! 1. **No `scf.for` anywhere.** The Rust loops in *this file* iterate tiles;
//!    the emitted IR is flat. The compiler loop IS the machine's timeline.
//! 2. **Every op carries `{slot, unit}`.** Time and place are part of the op.
//!    The portable `tile.*` IR hid both; on this architecture that hiding is
//!    the entire problem.
//! 3. **One memory space.** Everything lives in SRAM (space 1). There is no
//!    HBM to hide behind — capacity is the constraint, not bandwidth.
//!
//! Production notes (post 3.2 has the full list):
//! * Real Groq schedules bank conflicts, network hops between chips, and
//!   double-buffered movers; slots here are a *teaching granularity* (one op
//!   ≈ one slot), not TSP cycles.
//! * The Groq compiler is proprietary C++; this binary imitates its *output
//!   shape*, it does not interface with it (see "Why C++ enters" in the post).
//! * Data-dependent control flow (TinyMoE's router) becomes predication or
//!   compile-time bucketing — see the illustrative snippet in the post.
//!
//! `cargo run --bin groq_lowering` prints before/after and the slot count.

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

/// GEMM shapes. 256³ with 128 tiles → 2×2×2 = 8 micro-GEMMs, which unroll to
/// a schedule of ~30 slots — small enough to read, large enough to show the
/// pattern. Real programs are the same idea at 10⁵–10⁷ slots.
const M: i64 = 256;
const N: i64 = 256;
const K: i64 = 256;
const TILE: i64 = 128;

/// One memory space: SRAM. There is no "0 = HBM" in the machine's steady
/// state — weights are loaded once at model load and then everything is
/// on-chip. The 230 MB-class question is *capacity* (does the model fit?),
/// which is a compile-time fact, just like everything else here.
const SRAM: &str = "1 : i32";

/// Functional units, as they appear in the schedule. The TSP's units are
/// narrow specialists that all run on the compiler's clock: movers feed,
/// matmuls multiply, vectors do everything else.
const UNIT_MOVER: &str = "mover";
const UNIT_MATMUL: &str = "matmul";
const UNIT_VECTOR: &str = "vector";

/// The schedule clock: our stand-in for TSP time. Each emitted op stamps the
/// current slot and advances it. This struct is the whole architecture in
/// twenty lines of Rust — *the compiler owns time*.
struct Schedule {
    clock: i64,
}

impl Schedule {
    fn new() -> Self {
        Self { clock: 0 }
    }

    /// Allocate the next time slot.
    fn slot(&mut self) -> i64 {
        let s = self.clock;
        self.clock += 1;
        s
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);

    // Phase 1 — tile-level input (unchanged across the whole series).
    let before = build_tile_level(&context)?;
    println!("// ---- before: tile-level IR (portable middle) ----");
    println!("{}", before.as_operation());
    assert!(before.as_operation().verify(), "tile-level module must verify");

    // Phase 2 — lower to a flat, slot-stamped schedule.
    let (mut after, slots) = build_lpu_schedule(&context)?;

    // Phase 3 — upstream passes (CSE still useful; there are no loops left
    // for a loop optimizer to love).
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(transform::create_canonicalizer_pass());
    pass_manager.add_pass(transform::create_cse_pass());
    pass_manager.run(&mut after)?;

    println!("\n// ---- after: lower_tile_gemm_to_lpu (a schedule) ----");
    println!("{}", after.as_operation());
    println!("// total slots = {slots}");
    assert!(after.as_operation().verify(), "schedule module must verify");

    Ok(())
}

/// memref type in SRAM: `memref<128x128xf32, 1 : i32>`.
fn memref(context: &Context, rows: i64, cols: i64) -> Result<Type, Box<dyn Error>> {
    Ok(Type::parse(
        context,
        &format!("memref<{rows}x{cols}xf32, {SRAM}>"),
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

/// Phase 1: `"tile.gemm"(%a, %b, %c) {m, n, k}` — identical shape of code to
/// 0.1, 1.2, and 2.2. The portable middle does not change when the target
/// does. (By the third target this is less a claim and more a running joke.)
fn build_tile_level(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let sram = memref(context, M, N)?;
    let index_type = Type::index(context);

    let module = Module::new(location);
    let block = Block::new(&[(sram, location), (sram, location), (sram, location)]);
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
        TypeAttribute::new(FunctionType::new(context, &[sram, sram, sram], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok(module)
}

/// Stamp `{slot = N : index, unit = "..."}` onto an op under construction.
/// On this architecture an unstamped op is an *undefined behavior* in the
/// philosophical sense: nobody knows when or where it runs.
fn stamped<'c>(
    context: &'c Context,
    builder: OperationBuilder<'c>,
    unit: &str,
    slot: i64,
) -> OperationBuilder<'c> {
    builder
        .add_attributes(&[
            (
                Identifier::new(context, "unit"),
                StringAttribute::new(context, unit).into(),
            ),
            (
                Identifier::new(context, "slot"),
                IntegerAttribute::new(slot, Type::index(context)).into(),
            ),
        ])
}

/// Emit one stamped op into the function body and advance the clock.
/// (Nothing consumes results; these ops act on buffers — the same
/// side-effecting style as `tile.mxm` / `neuron.matmul` before them.)
#[allow(clippy::too_many_arguments)]
fn emit(
    context: &Context,
    location: Location,
    block: &Block<'_, '_>,
    schedule: &mut Schedule,
    unit: &str,
    name: &str,
    operands: &[Value],
    extra: Option<(&str, &str)>,
) -> Result<(), Box<dyn Error>> {
    let mut builder = OperationBuilder::new(name, location).add_operands(operands);
    if let Some((key, value)) = extra {
        builder = builder.add_attributes(&[(
            Identifier::new(context, key),
            StringAttribute::new(context, value).into(),
        )]);
    }
    let slot = schedule.slot();
    block.append_operation(stamped(context, builder, unit, slot).build()?)?;
    Ok(())
}

/// Phase 2: the schedule.
///
/// The emission *is* the lesson: Rust loops walk (ti, tj, tk) and emit a FLAT
/// sequence of stamped ops. There is no scf.for in the output because there is
/// no loop on the machine — a TSP "loop" is just the compiler repeating a
/// pattern of slots. The K dimension, the tile dimension, all of it unrolls
/// into time.
fn build_lpu_schedule(context: &Context) -> Result<(Module, i64), Box<dyn Error>> {
    let location = Location::unknown(context);
    let sram = memref(context, M, N)?;
    let tile = memref(context, TILE, TILE)?;

    let module = Module::new(location);
    let block = Block::new(&[(sram, location), (sram, location), (sram, location)]);
    let a_sram: Value = block.argument(0)?.into();
    let b_sram: Value = block.argument(1)?.into();
    let c_sram: Value = block.argument(2)?.into();

    // Indices used by the mover ops (which locate tiles inside the big
    // SRAM matrices). Plain constants — the only "control" left in the IR.
    // (There are no loop bounds to materialize: there are no loops.)

    // Scratch locations: two operand latches and the accumulator tile, all
    // SRAM. On the TSP, "where" is a bank/address the compiler picked; our
    // alloc stands in for that placement.
    let alloc = |result_type: Type| -> Result<Value, Box<dyn Error>> {
        let op = OperationBuilder::new("memref.alloc", location)
            .add_results(&[result_type])
            .build()?;
        Ok(block.append_operation(op)?.result(0).unwrap().into())
    };
    let a_tile = alloc(tile)?;
    let b_tile = alloc(tile)?;
    let c_tile = alloc(tile)?;

    let mut schedule = Schedule::new();

    // ---- The schedule body: loops in Rust, flat ops in the IR ----
    for ti in 0..M / TILE {
        for tj in 0..N / TILE {
            let si = index_const(context, &block, location, ti * TILE)?;
            let sj = index_const(context, &block, location, tj * TILE)?;

            // Movers stage the C accumulator region… on a real TSP the fill
            // is a vector-unit broadcast; keep it as one slot.
            emit(
                context,
                location,
                &block,
                &mut schedule,
                UNIT_VECTOR,
                "groq.fill",
                &[c_tile],
                None,
            )?;

            for tk in 0..K / TILE {
                let sk = index_const(context, &block, location, tk * TILE)?;

                // Two mover slots: fetch A[ti, tk] and B[tk, tj] tiles into
                // the operand latches. On TSP these are memory-mover units
                // pulling from SRAM banks on a fixed schedule — the whole
                // reason double-buffering "just happens" (the schedule says
                // so), not because hardware hides latency.
                emit(
                    context,
                    location,
                    &block,
                    &mut schedule,
                    UNIT_MOVER,
                    "groq.move",
                    &[a_sram, si, sk, a_tile],
                    Some(("direction", "sram_to_tile")),
                )?;
                emit(
                    context,
                    location,
                    &block,
                    &mut schedule,
                    UNIT_MOVER,
                    "groq.move",
                    &[b_sram, sk, sj, b_tile],
                    Some(("direction", "sram_to_tile")),
                )?;

                // One matmul slot: 128×128×128 on the matmul units,
                // accumulating into the C tile. Everything upstream existed
                // to make this slot legal at this time.
                emit(
                    context,
                    location,
                    &block,
                    &mut schedule,
                    UNIT_MATMUL,
                    "groq.matmul",
                    &[c_tile, a_tile, b_tile],
                    None,
                )?;
            }

            // Epilogue slot (vector units): TinyMoE kernel 4 would fuse here.
            emit(
                context,
                location,
                &block,
                &mut schedule,
                UNIT_VECTOR,
                "groq.epilogue",
                &[c_tile],
                Some(("op", "silu")),
            )?;

            // Mover slot: write the finished tile back.
            emit(
                context,
                location,
                &block,
                &mut schedule,
                UNIT_MOVER,
                "groq.move",
                &[c_tile, si, sj, c_sram],
                Some(("direction", "tile_to_sram")),
            )?;
        }
    }

    // Note what is NOT here: scf.for, scf.yield, any notion of "loop".
    // The function body is a timeline.
    block.append_operation(func::r#return(&[], location))?;

    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "expert_gemm"),
        TypeAttribute::new(FunctionType::new(context, &[sram, sram, sram], &[]).into()),
        region,
        &[],
        location,
    ))?;

    Ok((module, schedule.clock))
}
