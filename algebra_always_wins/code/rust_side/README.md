# lower — the post's descent, as a crate

Drives one `linalg.matmul` from "named structured op" to "LLVM dialect", dumping the IR
at every rung into `out/`. The dumps are the source for the post's descent section;
nothing in the post's IR listings is hand-typed that could be generated.

## Requirements

- Rust (any recent stable)
- **LLVM/MLIR 23** — `melior 0.28.x` depends on `mlir-sys 230.x`, which links MLIR 23.
  macOS: `brew install llvm@23`; Linux: `apt.llvm.org`. Point the build at it:

```bash
export MLIR_SYS_230_PREFIX=$(llvm-config --prefix)   # or /usr/lib/llvm-23
export LLVM_SYS_230_PREFIX=$MLIR_SYS_230_PREFIX
cargo run --release
```

This is the first place the post's thesis bites the author: **the binding is pinned to a
toolchain**. melior moves with upstream MLIR, mlir-sys moves with LLVM, and a pipeline
that runs on one pair may not run on the next. Dialects are stable; pipelines are not.

## What the crate reaches, and where its authority ends

| Rung | Pass | Reached from Rust? |
|---|---|---|
| 0 named | — (input) | ✅ |
| 1 generalized | `linalg-generalize-named-ops` | ✅ |
| 2 bufferized | `one-shot-bufferize` | ✅ |
| 3 scalar loops | `convert-linalg-to-loops` | ✅ |
| 4 LLVM dialect | `convert-{scf,arith,index,func,cf,memref}-to-llvm` | ✅ |
| 5 `vector.contract` | `transform.structured.tile_using_forall` + `vectorize` | ⚠️ needs a schedule |
| 6 `nvgpu.mma.sync` | `convert-vector-to-gpu{use-nvgpu=true}` | ⚠️ needs a schedule |
| 7 PTX / SASS | `mlir-translate`, `llc -march=nvptx64`, `ptxas`, `cuobjdump` | ❌ outside MLIR |

Rungs 5–6 require *decisions* (tile sizes, thread mapping, which levels go to shared
memory) that MLIR expresses as a transform-dialect **script**, not as flags. Melior 0.28
exposes `transform_dialect::create_transform_interpreter_pass()`, so you can run such a
script from Rust — but you have to write it. `fixtures/05-vector.mlir` and
`fixtures/06-mma.mlir` show what the far side looks like.

An honest note on scope: this crate stops where the *C API's* vocabulary stops. That is
the same phenomenon the post is about, one level up. Every tool has a cutoff.

## The C++-side command lines (if the binding disagrees with your LLVM)

```bash
# the five rungs the crate runs
mlir-opt kernels.mlir --pass-pipeline='builtin.module(linalg-generalize-named-ops,canonicalize,cse)'
mlir-opt ... --pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})'
mlir-opt ... --pass-pipeline='builtin.module(convert-linalg-to-loops)'
mlir-opt ... --pass-pipeline='builtin.module(convert-scf-to-cf,convert-index-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'

# rungs 5–6: a transform-dialect schedule, then the GPU dialogs
mlir-opt matmul.mlir --transform-interpreter \
  --pass-pipeline='builtin.module(func.func(convert-vector-to-gpu{use-nvgpu=true},convert-nvgpu-to-nvvm))'
```

## Why the input is Linalg and not StableHLO

StableHLO is an **out-of-tree** dialect. Upstream MLIR — which is what `mlir-sys`
links — cannot parse it, let alone lower it, without the StableHLO project's own
libraries. In practice you get StableHLO from `torch.export` + `torch-mlir`, or from
IREE's importer. So this driver picks up exactly one rung below where Torch-MLIR hands
off, and the post says so rather than pretending otherwise.
