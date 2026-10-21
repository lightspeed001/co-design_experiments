//! portable_audit — companion to post **7 "What the Portable IR Cannot Promise"**.
//!
//! The six lowerers in `src/bin/` are an accidental controlled experiment:
//! same input (`tile.gemm`), same driver (build → lower → canonicalize + CSE
//! → verify), same language (Rust + Melior) — six different machines. This
//! binary re-runs the *portable* half of that experiment and prints the
//! *shape fingerprint* of each machine: the five decisions every backend
//! re-learns, transcribed from the six files' code.
//!
//! ```text
//! // ---- the claim, testable ----
//! // portable across all six:
//! //   - build_tile_level()  (byte-identical in every binary)
//! //   - the pass pipeline   (canonicalize + CSE)
//! //   - the verifier        (MLIR's structural verify)
//! //   - the cast's 3 ROLES  (move / compute / coordinate)
//! // re-learned per machine:
//! //   loop fate · residency carrier · schedule carrier · sync · the leak
//! ```
//!
//! The fingerprint table below is transcribed by hand from the lowerers
//! (not scraped) so the post and this binary argue from one source.
//!
//! `cargo run --bin portable_audit` prints the middle + the fingerprints.

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

/// The portable middle's shapes — identical to all six siblings, on purpose.
const M: i64 = 256;
const N: i64 = 256;
const K: i64 = 256;

/// One machine's fingerprint: what its after-shape looks like and which
/// carrier carries each decision the middle cannot express.
struct Shape {
    post: &'static str,
    target: &'static str,
    after_shape: &'static str,
    loop_fate: &'static str,
    residency: &'static str,
    sync: &'static str,
    cast: &'static str,
}

/// Transcribed from the six lowerers + Maia's scorecard (5.1 was product-only:
/// an internal SDK leaves nothing to lower *to* — which is itself a fingerprint).
const SHAPES: [Shape; 7] = [
    Shape {
        post: "1.2",
        target: "TPU",
        after_shape: "one fat tile op",
        loop_fate: "kept (K panels)",
        residency: "spaces 0/1 (vec/scalar)",
        sync: "loop order + tile.fence",
        cast: "dma/tensor/vector",
    },
    Shape {
        post: "2.2",
        target: "Trainium",
        after_shape: "cast + second memory",
        loop_fate: "kept (K panels)",
        residency: "spaces 1/2 (SBUF/PSUM)",
        sync: "engine order",
        cast: "dma/tensor/vector",
    },
    Shape {
        post: "3.2",
        target: "Groq",
        after_shape: "flat slot-stamped schedule",
        loop_fate: "deleted -> time",
        residency: "SRAM slots (flat)",
        sync: "slot order IS the sync",
        cast: "mover/matmul/vector",
    },
    Shape {
        post: "4.2",
        target: "Tenstorrent",
        after_shape: "firmware grid + NoC",
        loop_fate: "kept (firmware loop)",
        residency: "spaces 0/1 + NoC traffic",
        sync: "ttix.barrier",
        cast: "dma/tensor/vector",
    },
    Shape {
        post: "5.1",
        target: "Maia",
        after_shape: "scored, not lowered",
        loop_fate: "(internal SDK)",
        residency: "HBM3e @ ~8 TB/s (reported)",
        sync: "(internal SDK)",
        cast: "(internal SDK)",
    },
    Shape {
        post: "5.2",
        target: "MTIA",
        after_shape: "minimal rewrite + gap fillers",
        loop_fate: "kept (Triton-shaped)",
        residency: "spaces 0/1 (LPDDR/SRAM)",
        sync: "mtia.ext_sync (extension)",
        cast: "dma/tensor/vector",
    },
    Shape {
        post: "6.2",
        target: "WSE",
        after_shape: "spatial placement + streaming",
        loop_fate: "spatialized (pipeline)",
        residency: "place attr (no spaces!)",
        sync: "dataflow triggers",
        cast: "stream/mac/fabric",
    },
];

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);

    // ---- the portable middle, built once ----
    // `build_tile_level` below is byte-identical to the copy in each of the
    // six lowerers — the strongest portability result of the series, and the
    // reason this audit can speak for all of them.
    let mut module = build_tile_level(&context)?;
    println!("// ---- the portable middle (shared by all six lowerers) ----");
    println!("{}", module.as_operation());

    // ---- the portable passes, run live ----
    // canonicalize + CSE: the only passes the six binaries all run. (No
    // target pass is portable: Groq's slot assignment, MTIA's gap fillers and
    // the wafer's placement are all bespoke builders, not passes.)
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(transform::create_canonicalizer_pass());
    pass_manager.add_pass(transform::create_cse_pass());
    pass_manager.run(&mut module)?;

    println!("\n// ---- after canonicalize + CSE (the portable pass pipeline) ----");
    println!("{}", module.as_operation());

    // ---- the portable verifier ----
    assert!(module.as_operation().verify(), "portable middle must verify");
    println!("// verified: ok");

    // ---- the shape fingerprints ----
    // Rows are padded to fixed widths but trimmed at the line end so the
    // dump in post 7 matches this output character-for-character (trailing
    // whitespace is the enemy of faithful dumps).
    println!("\n// ---- shape fingerprint: what every machine made us re-learn ----");
    println!(
        "{}",
        format!(
            "{:<6} {:<12} {:<32} {:<24} {:<28} {:<28} {:<22}",
            "post", "target", "after-shape", "loop fate", "residency carrier", "sync carrier", "cast"
        )
        .trim_end()
    );
    for shape in SHAPES {
        println!(
            "{}",
            format!(
                "{:<6} {:<12} {:<32} {:<24} {:<28} {:<28} {:<22}",
                shape.post,
                shape.target,
                shape.after_shape,
                shape.loop_fate,
                shape.residency,
                shape.sync,
                shape.cast
            )
            .trim_end()
        );
    }

    println!("\n// the middle promised: tiles, data movement, the cast's roles,");
    println!("//   structural verification, composable passes.");
    println!("// the middle cannot promise: where data lives, who runs when,");
    println!("//   when things are done, how the loop lives, what the machine is for.");

    Ok(())
}

/// The portable middle: `"tile.gemm"(%a, %b, %c) {m, n, k}`. Copied verbatim
/// into every `src/bin/*_lowering.rs` — post 7's thesis in 30 lines of code.
fn build_tile_level(context: &Context) -> Result<Module, Box<dyn Error>> {
    let location = Location::unknown(context);
    let lpddr = memref(context, M, N, "0 : i32")?;
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

/// memref with a memory space: `memref<256x256xf32, 0 : i32>`.
fn memref(context: &Context, rows: i64, cols: i64, space: &str) -> Result<Type, Box<dyn Error>> {
    Ok(Type::parse(
        context,
        &format!("memref<{rows}x{cols}xf32, {space}>"),
    )?)
}
