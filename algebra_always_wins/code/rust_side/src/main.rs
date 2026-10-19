//! lower.rs — walk one matmul down the ladder, one rung at a time.
//!
//! The design of this file is the argument of the post, expressed as code: a lowering
//! is not a single transformation, it is a *sequence of vocabularies*, and at each step
//! the set of things you are allowed to say shrinks. So instead of one big pass
//! pipeline we run one rung, print the module, and go again. The `out/` directory is
//! the post's figure 2.
//!
//! Build (needs LLVM/MLIR 23; see README.md):
//!     MLIR_SYS_230_PREFIX=$(llvm-config --prefix) cargo run --release
//!
//! Output:
//!     out/00-input.mlir ... out/04-llvm.mlir   plus a per-rung dialect histogram

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use melior::{
    dialect::DialectRegistry,
    ir::Module,
    pass::PassManager,
    utility::{parse_pass_pipeline, register_all_dialects, register_all_passes},
    Context,
};

/// One rung of the ladder.
///
/// `pipeline` is spelled exactly as it would be on the command line, so every rung in
/// this file is also a one-liner you can reproduce with `mlir-opt` if the Rust binding
/// ever disagrees with your LLVM version. That escape hatch matters: dialects are
/// stable, pipelines are not.
struct Rung {
    /// Filename stem for the dumped IR.
    slug: &'static str,
    /// What is being destroyed at this rung — the only column that matters.
    casualty: &'static str,
    pipeline: &'static str,
}

const RUNGS: &[Rung] = &[
    // ---------------------------------------------------------------------------------
    // RUNG 1 — the named op dies.
    //
    // `linalg.matmul` becomes a `linalg.generic`: three affine maps, two iterator
    // kinds, and a body containing literally `a*b+acc`. From here on, no pass in the
    // entire compiler knows that a matrix multiply is happening. It knows about an
    // index space with two parallel dimensions and one reduction dimension, and that
    // is strictly less information — but it is also strictly more *usable*, because
    // every op in the program now has the same shape, so the same machinery can
    // optimize all of them.
    // ---------------------------------------------------------------------------------
    Rung {
        slug: "01-generalized",
        casualty: "'matmul' — the operation's identity",
        pipeline: "builtin.module(linalg-generalize-named-ops,canonicalize,cse)",
    },
    // ---------------------------------------------------------------------------------
    // RUNG 2 — tensors become buffers.
    //
    // Values stop being values. After one-shot bufferization every tensor is a memref:
    // a pointer, a shape, strides and an offset. Two things die here at once — the
    // freedom to have an unbounded working set (there is now an allocation), and the
    // freedom to pretend that layout is a detail (strides are in the type).
    // ---------------------------------------------------------------------------------
    Rung {
        slug: "02-bufferized",
        casualty: "value semantics — tensors become addresses",
        pipeline: "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true},canonicalize,cse)",
    },
    // ---------------------------------------------------------------------------------
    // RUNG 3 — the structured op dies.
    //
    // The affine maps are gone. What is left is `scf.for` over three induction
    // variables with `memref.load` / `memref.store` in the body. This is the exact
    // moment the post calls out: the program stops being a *description of an algebra*
    // and becomes a *description of a traversal*. If you want to know where your
    // performance went, it went here, into a choice of traversal order.
    // ---------------------------------------------------------------------------------
    Rung {
        slug: "03-loops",
        casualty: "'structured' — index maps become induction variables",
        pipeline: "builtin.module(convert-linalg-to-loops,canonicalize,cse)",
    },
    // ---------------------------------------------------------------------------------
    // RUNG 4 — the dialect dies.
    //
    // Everything left is LLVM dialect: memrefs become `llvm.extractvalue` descriptor
    // arithmetic, loops become basic blocks, and the multiply-add becomes an `llvm.fmul`
    // plus an `llvm.fadd`. Below this line there is no algebra at all — only load,
    // store, branch, and a register allocator. `reconcile-unrealized-casts` is the
    // bookkeeping that removes the scaffolding the conversion passes leave behind.
    //
    // (Rungs 5–6 — vector.contract and nvgpu.mma.sync — need scheduling decisions that
    // the C API does not surface in melior 0.28. See README.md; the fixtures in
    // fixtures/ show the IR, and the same file documents the transform-dialect script
    // that produces it.)
    // ---------------------------------------------------------------------------------
    Rung {
        slug: "04-llvm",
        casualty: "MLIR itself — only machine-level ops remain",
        pipeline: "builtin.module(convert-scf-to-cf,convert-index-to-llvm,convert-arith-to-llvm,\
                   finalize-memref-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,\
                   reconcile-unrealized-casts)",
    },
];

fn main() -> Result<(), Box<dyn Error>> {
    // Register every pass and dialect this MLIR build knows about. If your build lacks
    // one (say, nvgpu), the failure is at `parse_pass_pipeline` time with a helpful
    // message, not a mysterious crash later.
    register_all_passes();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    // `true` = enable multithreading inside MLIR. Irrelevant for a 60-line module, and
    // essential once you actually try this on a model.
    let context = Context::new_with_registry(&registry, true);
    context.load_all_available_dialects();

    let source = fs::read_to_string("kernels.mlir")?;
    let mut module = Module::parse(&context, &source).ok_or("kernels.mlir failed to parse")?;

    let out = Path::new("out");
    fs::create_dir_all(out)?;

    let mut rung_rows: Vec<(String, usize, usize, BTreeMap<String, usize>)> = Vec::new();
    rung_rows.push(snapshot(&module, out, "00-input", "as written")?);

    for rung in RUNGS {
        // A fresh pass manager per rung: the point is to observe the module *between*
        // rungs. A single pipeline would be faster and would tell you nothing.
        let pm = PassManager::new(&context);
        parse_pass_pipeline(pm.as_operation_pass_manager(), rung.pipeline)?;
        pm.run(&mut module)?;

        let row = snapshot(&module, out, rung.slug, rung.casualty)?;
        rung_rows.push(row);
    }

    // The table the post quotes: the module gets *longer* as it descends (more
    // instructions, more dialects) while the number of distinct concepts gets smaller.
    // Compression is not measured in line count.
    println!("\n{:<18} {:>6} {:>8}  {}", "rung", "lines", "ops", "vocabulary (ops by dialect)");
    println!("{}", "-".repeat(78));
    for (name, lines, ops, hist) in &rung_rows {
        let vocab: Vec<String> = hist.iter().map(|(d, n)| format!("{d}({n})")).collect();
        println!("{:<18} {:>6} {:>8}  {}", name, lines, ops, vocab.join(" "));
    }

    Ok(())
}

/// Dump the module, and measure it in a way that is honest about what changed.
///
/// Counting *ops by dialect* rather than lines is deliberate: line count goes up as you
/// descend (LLVM IR is verbose), while the number of distinct dialects you need to know
/// about stays small and the number of distinct *ideas* collapses.
fn snapshot(
    module: &Module,
    dir: &Path,
    slug: &str,
    note: &str,
) -> Result<(String, usize, usize, BTreeMap<String, usize>), Box<dyn Error>> {
    let text = module.as_operation().to_string();
    let path = dir.join(format!("{slug}.mlir"));
    fs::write(&path, &text)?;

    let lines = text.lines().count();
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut ops = 0usize;
    for line in text.lines() {
        let line = line.trim_start();
        // Skip the pretty-printer's continuations and block labels; keep the op lines.
        let Some(head) = line.split_whitespace().next() else { continue };
        let Some((dialect, _)) = head.split_once('.') else { continue };
        if !head.ends_with(':') && dialect.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
            *hist.entry(dialect.to_string()).or_default() += 1;
            ops += 1;
        }
    }

    println!("· {:<18} {:<44} {} lines", slug, note, lines);
    Ok((slug.to_string(), lines, ops, hist))
}
