# tinymoe-lowerer

Companion crate for post **0.1**. Builds the TinyMoE expert GEMM as tile-level
MLIR (`tile.gemm`), then lowers it to `scf`/`arith`/`memref` loops in pure Rust
via [Melior](https://github.com/mlir-rs/melior) — and runs upstream MLIR passes
(canonicalizer, CSE) from Rust.

## Requirements

- Rust (stable, edition 2021+)
- **LLVM/MLIR 23** (melior 0.27 links the MLIR C API):
  - macOS: `brew install llvm@23`
  - Ubuntu: `apt install llvm-23-dev libmlir-23-dev mlir-23-tools` (package
    names vary by distro version)
  - `mlir-sys` (melior's FFI layer) discovers MLIR via `llvm-config`. If it
    can't find it, set the prefix env var documented in the
    [mlir-sys](https://github.com/mlir-rs/mlir-sys) README (e.g.
    `MLIR_SYS_230_PREFIX=/path/to/llvm-23`) and put `llvm-config` on `PATH`.

No CUDA, no C++, no code generation steps — the whole lowering is Rust.

## Run

```sh
# post 0.1 — generic lowering: tile.gemm → scf/arith/memref loops
cargo run

# post 1.2 — TPU-shaped lowering: tile.gemm → HBM/VMEM staging + tile.mxm
cargo run --bin tpu_lowering

# post 2.2 — Trainium-shaped lowering: tile.gemm → SBUF/PSUM + engine cast
cargo run --bin trainium_lowering

# post 3.2 — LPU-shaped lowering: tile.gemm → a flat, cycle-stamped timetable
cargo run --bin groq_lowering

# post 4.2 — Tensix-shaped lowering: tile.gemm → a core grid with NoC + barriers
cargo run --bin tenstorrent_lowering

# post 5.2 — MTIA-shaped lowering: tile.gemm → minimal rewrite + extensions
cargo run --bin mtia_lowering
cargo run --bin cerebras_lowering
cargo run --bin portable_audit
```

Expected output for each: IR dumps — the tile-level module, the lowered module,
and the same module after `canonicalize` + `cse`. Each post's binary is
self-contained on purpose (blog code you can read top-to-bottom); shapes are
small so the dumps fit on a screen (edit the constants at the top of each file).

## Binaries

| Binary | Post | What it shows |
|---|---|---|
| `src/main.rs` | 0.1 | the portable middle lowered to generic loops |
| `src/bin/tpu_lowering.rs` | 1.2 | the same input lowered to a TPU-shaped program (VMEM memory spaces, DMA staging, `tile.mxm`, epilogue slot) |
| `src/bin/trainium_lowering.rs` | 2.2 | the same input lowered to a Trainium-shaped program (SBUF + PSUM memory spaces, explicit engine cast, `neuron.matmul` accumulating in PSUM) |
| `src/bin/groq_lowering.rs` | 3.2 | the same input lowered to an LPU-shaped program: **loops dissolve** into a flat schedule where every op is stamped with a time slot and a functional unit |
| `src/bin/tenstorrent_lowering.rs` | 4.2 | the same input lowered to a Tensix-shaped program: **the loops return** as firmware on a core grid, with explicit NoC moves and barriers |
| `src/bin/mtia_lowering.rs` | 5.2 | the same input lowered to an MTIA-shaped program: the **minimal-rewrite** shape — the tile program survives, gaps are filled with *mapping math* (linear `program_id` → 2D grid) and *language extensions* (`mtia.ext_*`) |
| `src/bin/cerebras_lowering.rs` | 6.2 | the same input lowered to a WSE-shaped program: the **spatial-placement** shape — the loop dissolves into a weight-streaming pipeline, every op carries a `place`, on-wafer types drop memory spaces |
| `src/bin/portable_audit.rs` | 7 | the portable middle built + verified live, plus the **shape fingerprint** table for all six lowerings (loop fate, residency, sync, cast) |

## Where this goes in the series

Post 0.1 explains the design (why an unregistered `tile.*` dialect, why bounds
are SSA operands vs attributes, why the same `GemmSpec` drives both builders).
Later posts extend this crate with **target hooks** — a target trait whose
implementations decide tile sizes, buffer scopes (HBM ↔ scratchpad), and
synchronization for each architecture's Part B.
