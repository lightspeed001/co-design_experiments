# "Algebra always wins" — structural plan

> **Status: drafted.** → [`post.md`](post.md) (publishable Markdown) ·
> [`algebra-always-wins.html`](algebra-always-wins.html) (self-contained, figures inlined).
> Artifacts in [`code/`](code) and [`figures/`](figures). This file is the plan the draft
> was built from; what changed during drafting is logged at the bottom.

## Draft log — what changed when the prose met the artifacts

1. **The 300× hinge replaces three separate arguments.** §4.2 now carries the entire
   "so what" with two numbers (AI = 1,365 vs AI = 1) instead of a list of consequences.
   The KV-cache, quantization and speculative-decoding points became one-sentence
   corollaries of the ridge rather than their own subsections.
2. **The descent stopped at four rungs in the code, not seven.** Melior walks
   generalize → bufferize → loops → LLVM. Rungs 5–6 need a *schedule* (transform-dialect
   script), not a flag, so they live in `fixtures/` and §3.7 now argues from that cutoff
   instead of hiding it. The limitation turned into a paragraph of evidence.
3. **StableHLO is out of reach from upstream MLIR**, so the crate starts at `linalg.matmul`
   and the post says why. Third, smaller "every tool has a cutoff" beat, kept because it
   is funny and true.
4. **The Triton kernel earned its place.** It was going to be a footnote; it became §5.1's
   primary evidence, because the monoid is *visible in the source* there in a way it isn't
   in a paper's pseudocode.
5. **The measured histogram moved earlier.** `torch.export` on one block gives 37 layout
   ops out of 83, which is a better opening for the descent than any amount of prose
   about layouts. It is now §3.2 and it is generated, not asserted.
6. **§7 got the sharpest line in the piece** ("our sloppy arithmetic is what licenses the
   substrate") and kept it. §8's inversion kept the second-sharpest ("because it isn't
   competing").


**Subtitle:** *How every architecture dissolves into matmuls, and why the substrate chose them*

---

## 0. The thesis, stated three ways

Write these down before any section, because the whole structure is just the argument that earns them.

**The one-liner (for the abstract):**
> Every level of the ML stack is an effective theory with a bounded domain of validity. Lowering is what happens at the boundary: the vocabulary of one level stops being meaningful and a smaller, harder vocabulary takes over. At the bottom of that flow there is a fixed point — an associative combine over a blocked index space — and the hardware is a machine built to perform reassociation at the highest rate per joule. So the architecture is not the invariant. The schedule is not the invariant either. The invariant is the algebra.

**The sharper, defensible claim (the one the post must actually defend):**
> Model definitions don't *reduce* to matmul because the math happens to be linear. They reduce to matmul because matmul is the intersection of (a) an associative operator, which grants the compiler freedom to reorder, (b) uniform dataflow, which grants the silicon dense locality, and (c) a shape that can be blocked, which converts bandwidth into arithmetic intensity. Any operator with those three properties dissolves the same way. Any operator missing one of them survives lowering as *something else* — traffic, a collective, or control flow.

**The claim you should explicitly refuse to make:**
> "Everything is just a matmul 🙂." False, and readers will punish it. Softmax, top-k, cumsum, gather/scatter, sampling, and dynamic shapes are not matmuls. The interesting content of the post is *what they turn into instead* — and why that residue is always bandwidth or control, never compute.

That third box is the credibility of the piece. Put the residue in the middle of the post, not in a hand-wave at the end.

---

## 1. Naming the metaphor, once, precisely

The "effective theory" framing is load-bearing, so use it exactly once, cleanly, then drop the physics costume and speak engineering. Four correspondences are worth spelling out, because they generate real structure rather than decoration:

| Physics | This post |
|---|---|
| Effective theory valid below a cutoff scale | Abstraction valid above a tile size / precision / shape assumption |
| Renormalization-group flow | The pass pipeline; each pass coarse-grains and forgets structure |
| Fixed point / universality class | The blocked loop nest (`linalg.matmul` → `vector.contract` → `mma.sync`) |
| Irrelevant vs. marginal operators | Architecture details vanish under lowering (irrelevant); **precision never vanishes** (marginal), and it's the only thing that changes the answer |

That last row is the best idea in the post and most people miss it. Say it in one line and let it pay off in §6:
> The architecture is an irrelevant operator: it disappears at every scale below the model. Precision is a marginal operator: it survives every pass, and it is the only thing that can change the result.

**Guardrail:** no more than ~120 words of physics total. One good analogy, used structurally, reads as rigor. Extended physics cosplay reads as a sophomore essay.

---

## 2. The hook (≈300 words)

Three candidates, in order of preference:

**(A) The profiler's view — recommended.**
Open with a trace, not a definition. "Here is one token of decode from a 7B transformer, as the profiler sees it." List the kernel families: a GEMM, a GEMV-ish batched GEMM, a fused elementwise/reduction kernel (RoPE + softmax + norm + activation), a copy (KV-cache write / embedding gather), a collective. Then: *there is no attention in this picture.* No "layer", no "head", no "block". The architecture you drew on the whiteboard is not in the trace. What's in the trace is five families of algebra and the traffic between them.
Why this works: it's empirical, it's falsifiable, and it hands the reader a command they can run tonight (`ncu`/`nsys` on their own model). Technical readers convert on "I can check this."

**(B) The scroll.** One line of PyTorch — `attn = softmax(Q @ K.T / sqrt(d)) @ V` — and the eleven successive IRs it becomes, printed one per screen, ending at `HMMA.16816.F32`. Compelling, but it front-loads detail before the reader knows why to care. Better as §4's spine than as the opening.

**(C) The physics cold open.** "A neural network is a coarse-grained description of a computation that has already happened. The coarse-graining has a direction, and at the bottom of it the description stops changing." Elegant, risky. If you use it, compress it to two sentences and get to (A) fast.

Whichever you pick, end the hook on the promise, not the thesis:
> This post follows one architecture all the way down, and reports what's left when the vocabulary runs out.

---

## 3. The descent (≈1,100 words) — the technical centerpiece

This is the section reviewers will cite, so it has to be *real*: actual IR, actual pass names, actual diff-looking transforms. Keep it under a third of the post or it becomes a dialect tour.

### 3.1 The ladder (one table, and it's the map for everything after)

| Level | Vocabulary | What the theory believes | What breaks it |
|---|---|---|---|
| Model (PyTorch/JAX) | tensors, modules, autograd, `attention` | composition is free; shapes are symbolic | tracing: control flow, aliasing, mutation, data-dependent shapes |
| Graph IR (Dynamo/FX, StableHLO) | `dot_general`, `reduce`, `exp` | ops are pure and total | fusion decisions, layout, guards/recompiles |
| Structured ops (Linalg on tensors) | `linalg.matmul`, `linalg.generic` | tensors are values with index semantics | the named op dissolves into index maps; nothing left to name |
| Loops & buffers (SCF/memref, affine) | `scf.forall`, `memref.subview`, `affine_map` | "tensor" means something | tensor → address arithmetic; layout becomes a type parameter |
| Vector/GPU (vector, nvgpu) | `vector.contract`, `nvgpu.mma.sync` | matmul is a unit | matmul becomes one instruction *shape*: m16n8k16 fragments |
| ISA (NVVM/PTX/SASS) | `mma.sync`, `wgmma`, `HMMA.16816` | the compiler still exists | scheduling: registers, occupancy, TMA, warp specialization |
| Silicon (tensor core / MXU / systolic array) | `D += A·B` on a fixed grid | — | the substrate's own limit: bytes/joule |

The value of this table is the third and fourth columns. **A stack walk is only interesting if every rung reports a casualty.** Without that, §3 is a Wikipedia summary of MLIR.

### 3.2 The three moments worth showing in full

Don't show all seven levels equally. Show three transformations and summarize the rest, because these three carry the argument:

1. **The named op dies.** `stablehlo.dot_general` → `linalg.matmul` → `linalg.generic` with three affine maps and a scalar body `yield a*b + acc`. This is the pivot of the whole post: at the moment "matmul" becomes "a loop nest with a multiply-add inside," the *architecture's last trace vanishes* and only an index space remains. Quote the `linalg.generic` body literally, three lines. Say: this is the last IR in which you can still see the model.

2. **Tiling, and why it's legal.** `scf.forall (%i, %j) in (M/128, N/128)` around a 128×128×32 block, with `memref.subview` + `vector.transfer_read`. Here's the sentence to write in bold: *the tiled op has the same index maps as the untiled op.* Tiling is a restatement of the algebra on blocks, and it's valid because the operator is associative — the same reason your pass pipeline is allowed to exist. Everything downstream (occupancy, SRAM reuse, tensor-core shapes) is a consequence of this one fact.

3. **The algebra gets a hardware quantum.** `vector.contract` with `indexing_maps = [(i,k),(k,j),(i,j)]`, `iterator_types = ["parallel","parallel","reduction"]` → `nvgpu.mma.sync {mma_shape = #nvgpu<[16,8,16]>}` → `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` → `HMMA.16816.F32` in SASS. One instruction, 2,048 multiply-accumulates. The abstraction hasn't been *reduced* to algebra; it has been *quantized* to the algebra the silicon can actually execute.

### 3.3 The Rust/Melior artifact

This is what makes the post yours instead of a Medium post. Ship a small crate that drives the actual pipeline and dumps the IR at each stage, so readers can reproduce the descent on their own op.

- `melior` (mlir-rs/melior) wraps the MLIR C API: `Context`, `Module`, `PassManager`, dialect registry. You can register StableHLO/Linalg/SCF/Vector/NVGPU and run the textual pipeline `mlir-opt` would run, then `print()` the module between stages.
- Pipeline that works today (CPU path and GPU path), roughly:
  `builtin.module(stablehlo-legalize-to-linalg, one-shot-bufferize, linalg-tile..., convert-linalg-to-loops)` for the legible version; then the GPU variant `… convert-vector-to-gpu, convert-gpu-to-nvvm, convert-nvvm-to-llvm`, `mlir-translate`, `llc -march=nvptx64`, `ptxas`.
- Practical notes worth a footnote: melior tracks upstream LLVM closely — pin `mlir-sys` to an LLVM version pair (220.x ↔ LLVM 22, etc.) or the build will drift under the reader. Use `enable_ir_printing`/`-mlir-print-ir-after-all` and capture the stages into a fixture directory so the post's code blocks are generated, not hand-typed.
- One honest caveat, included in the post: **the pipeline is not portable.** Dialects are stable; pass pipelines are not, and a sequence tuned for Torch-MLIR will not run unmodified through IREE. That fragility is itself evidence for the thesis — the "effective theory" boundary is an *engineering* boundary as much as a semantic one.

**Deliverables:** `lower.rs` (~150 lines), a `fixtures/` dir of the six IR stages, and one figure that shows the same 40 lines of StableHLO collapsing to 9 lines of `vector.contract` + `nvvm`. The shrinking line count *is* the argument: lowering is lossy compression, and what survives is what the silicon can use.

---

## 4. Why the fixed point is GEMM (≈800 words)

This is the section the subtitle promises. It must be an argument, not an appreciation. Three moves:

### 4.1 The three properties (from §0's sharpened claim)

- **Associativity** → the compiler's license to tile, split-K, reorder, pipeline, and fuse. Without it, none of §3.2's transformations are legal rewrites.
- **Uniform dataflow** → no branches, no data dependence on values, static shapes at the level of the inner loop. This is what makes dense locality *buildable*: systolic arrays communicate only with nearest neighbors (Kung & Leiserson, 1978); tensor cores read fragments from registers with a fixed pattern.
- **Blockability** → the operator's index space factors, and each factor can be loaded once and reused O(block) times. This is the only mechanism that converts bandwidth into throughput.

Then the line that pays off the whole piece:
> Multiplication isn't the point. **Reassociation is the point**, and multiplication is the cheapest operator that comes with a lot of it.

### 4.2 The roofline, with numbers (this is the "so what")

Use H100 SXM dense figures and show the two regimes. State them as the machine's *balance point*:

- BF16 dense ≈ 989 TFLOP/s; HBM3 ≈ 3.35 TB/s → ridge ≈ **295 FLOP/byte**.
- **4096³ GEMM:** 137 GFLOP over ~100 MB → AI ≈ 1,370 → ~4.6× above the ridge → **compute-bound**. Time: ~0.14 ms of math vs ~0.03 ms of traffic.
- **4096×4096 GEMV** (the decode-time projection / LM head): 33.5 MFLOP over 33.5 MB → AI = **1** → ~300× below the ridge → **memory-bound**. Time: ~10 µs of traffic vs ~34 ns of math.

That's a 300× swing in what the machine is *for*, produced by one change in shape. Make this the hinge of the post:

> **GEMM is where algebra wins on compute. GEMV is where algebra wins on *bytes*.** Same operator, same associativity; different side of the ridge. The reason batching exists, the reason the KV cache exists, the reason decode is bandwidth-starved while prefill is FLOP-starved, the reason 4-bit weights help at batch 1 and stop helping at batch 256 — all of it is this single ratio.

This is the section that makes the essay *useful* rather than merely true. Most "everything is matmul" posts stop at the observation; this one turns it into a lens for reading the whole systems literature. (If you want one more paragraph: put quantization, speculative decoding, and prefill/decode disaggregation in a short list, each mapped back to the ridge. One line each. Don't let it become a survey.)

### 4.3 Why the substrate "chose" them — and why that phrasing is backwards

Kill the teleology here, deliberately, because it's the most interesting turn available:
> The substrate didn't choose matmul. Matmul is what a substrate *is* once it gets fast enough. Every order-of-magnitude gain since 1978 has been another way to perform the same multiply-accumulate: systolic arrays → BLAS level 3 → blocked Goto/BLIS kernels → cuDNN's implicit GEMM → the TPU's 128×128 MXU → Volta's tensor cores → Hopper's `wgmma` + TMA → Blackwell's fp4 path. The architectures were proposed by people; the substrate accepted only the ones that could be expressed as a blocked contraction. That's not a preference. It's a filter.

---

## 5. Three dissolves (≈700 words)

Case studies, and each must show a *different* escape hatch, or they're just three examples of the same point. Order them by increasing resistance:

1. **Attention — the nonlinearity that turned out to be schedulable.**
   The one architecture explicitly designed to be non-algebraic: softmax is not linear, and naive attention materializes an N² matrix, which is a memory-traffic disaster, not a compute one. FlashAttention is the case study because it *doesn't* remove the nonlinearity — it proves the nonlinearity is a **monoid** (max, and log-sum-exp, are associative combines), so it can be tiled exactly like a reduction and fused between the two GEMMs. The reader's takeaway: *softmax didn't resist algebra; it revealed that the winning algebra is the semiring, not just linear algebra.* Attention's FLOPs stay ~2 GEMMs + a small correction; the win is entirely in not writing 4 GB of intermediates.

2. **Convolution — three normal forms for one op, chosen by the substrate.**
   Direct convolution → **implicit GEMM** (Toeplitz structure hidden inside the index maps; the "conv" was always a matmul wearing a costume) → **Winograd** (fewer multiplies — algebra genuinely winning on operation count) → **FFT** (asymptotically better, usually loses on real hardware). The punchline writes itself: Winograd reduces arithmetic and *loses* below a certain size because it increases traffic; the substrate gets a veto over mathematics. A beautiful, concrete counterexample to pure-algebraic reasoning, which is why it belongs in a post titled "algebra always wins."

3. **MoE / quantization / SSMs — the frontier where the algebra fragments.**
   Grouped GEMM for experts; gather + dequant + GEMM for 4-bit weights; the chunked scan for Mamba (a recurrence that becomes a *parallel prefix*, then matmuls). Show the pattern: each of these is *still* algebra, but the blocks get smaller and the intensity drops, so the schedule — not the math — becomes the whole problem. This is the honest edge of the thesis: the algebra doesn't get simpler at the frontier, it gets *worse-conditioned*.

---

## 6. The residue: what refuses to dissolve (≈600 words)

Non-negotiable section. Put it *after* the case studies so it reads as rigor, not hedging. A short taxonomy, each with its escape:

- **Memory-bound elementwise & reductions** (residual adds, norms, activations, masks). FLOPs are free; bytes are not. The escape is *fusion*, i.e., a scheduling win. Fuse-ability is why compilers exist.
- **Associative but non-linear** (softmax, cumsum, scan, top-k's partial sorts). Escapes into monoids and parallel prefix. Algebra still wins — a different algebra.
- **Data-dependent control** (routing, dynamic shapes, sampling, tokenization). Here lowering stops being a total function: the graph breaks, guards appear, recompiles fire. This is the true boundary of the "effective theory," and the place to say so plainly.
- **Collectives** (all-reduce, all-gather, reduce-scatter). Extend the tower *upward*, not just down: sharding is a change of basis, tensor parallelism is a block decomposition of the same matmul, ring all-reduce is a scan over ranks. At cluster scale, the "GEMM" is bandwidth and the algebra is the *communication pattern*. One paragraph — this is where a lot of your readers live.
- **The kernel zoo as evidence.** Compile any modern model and you get ~5–8 kernel families out of a space of thousands of possible ops. That collapse *is* the universality class. If the thesis is right, this is what you should see — and you do.

Then the honest summary line:
> Compute dissolves. Traffic and control do not. Every optimization that has ever made a model 10× faster was an attack on one of those two, dressed up as an attack on the other.

---

## 7. The error term (≈450 words) — the best section nobody writes

The paradox: **the tower is only standing because floating point is wrong.**

- In ℝ, matmul is associative, so tiling/reassociation is a legal rewrite. In IEEE-754 it isn't. If ML required bit-exact results, every transformation in §3.2 would be an illegal rewrite and you would lose essentially all of the performance.
- So the whole edifice rests on a deliberate lie — *that a·b+c is associative* — plus a community-wide tolerance for the error it introduces. `split-K` changes the summation tree, TF32 changes the multiplicands, bf16 accumulation changes the result, fp8 with per-tensor scaling changes everything. Determinism is a property of the **schedule**, not of the mathematics.
- Therefore: reproducibility bugs are not bugs at the boundary between software and hardware; they *are* the boundary. They are where the effective theory lets you see its own error term.
- And this is where the marginal operator pays off: the architecture vanished at `linalg.generic`, but precision never did. It is the one quantity that survives every pass and changes the answer at every level.

Line to end the section on: *The substrate doesn't just tolerate our sloppy arithmetic. Our sloppy arithmetic is what licenses the substrate.*

---

## 8. The inversion (≈350 words) — the ending the subtitle is pointing at

Return to the hook's profiler trace. Nothing has changed: still five kernel families. But the reader now knows why.

- **The architecture is a proposal. The substrate is a constraint.** New architectures succeed roughly in proportion to how expressible they are as blocked contractions plus a small, schedulable residue. This is not a law of nature; it's a law of silicon economics, and it would change if the substrate changed (photonic crossbars, analog in-memory multiply, wafer-scale locality — mention one, don't speculate).
- **The model is a hypothesis written in a language with one verb.** That's not a diminishment. It's why a 1969 matrix multiply and a 2026 frontier model can be optimized by the same person with the same mental model.
- Close on the inversion, not on the metaphor:

> The architectures keep arriving, and the algebra keeps absorbing them. Not because linear algebra is the secret truth of intelligence — but because we built a machine that can only be convinced by it, and then built the models to fit the machine. Algebra always wins because it isn't competing.

**Optional coda (150 words, or a sidebar):** four practitioner heuristics. (1) Count FLOPs and bytes before you optimize anything; find which side of the ridge you're on. (2) If your op isn't a contraction, pick one of three doors: fuse it, reassociate it into a monoid, or batch it. If it opens none, admit it's a scheduling problem. (3) Track your "dissolve ratio" — the share of runtime in GEMM vs. everything else — it's a maturity diagnostic for any model+harness pair. (4) Read the IR, not the paper: the paper describes the architecture, the IR describes what the machine understood.

---

## 9. Figures & artifacts

| # | Figure | Why it's load-bearing |
|---|---|---|
| 1 | The ladder (the §3.1 table as a diagram, vocabularies on the left, casualties on the right) | the map; readers screenshot it |
| 2 | The IR column: 6 stages, one above the other, line counts shrinking | *shows* dissolution, doesn't assert it |
| 3 | Roofline with both points plotted (GEMM above the ridge, GEMV far below) | the "so what" |
| 4 | FlashAttention as dataflow: two GEMMs with a monoid sandwich, tiles circled | makes the semiring argument visible |
| 5 | Kernel-family collapse: ops → 5–8 families | the empirical receipt |

Code artifacts: `lower.rs` + `fixtures/` (IR stages, generated not typed), a 20-line `torch` script that prints the Dynamo graph and the inductor kernel, and the exact `ncu` invocation used for the hook's trace. Readers reproducing your numbers is worth more than any argument.

---

## 10. Pacing and cuts

Budget: **4,500–5,200 words**, or split into two posts at §5 if it runs long (`I. The descent` / `II. Why the substrate chose them` — the seam is clean).

- §3 must stay ≤ 1/3. The classic failure of this genre is that the IR walk eats the essay.
- If forced to cut: drop one of the three dissolves (keep attention + convolution), and compress §5 into §6's taxonomy. **Never cut §6 or §7** — they're the difference between an essay and a manifesto.
- If forced to add: more numbers in §4.2. Readers trust arithmetic.

**Three failure modes to actively avoid:** (a) the tautology — "everything is matmul because matmul is linear algebra," which is true and says nothing; (b) the dialect tour — a survey of MLIR with a thesis bolted on; (c) uninterrogated physics — the EFT metaphor is a *scaffold for the argument*, and the moment it starts generating sentences the engineering can't cash, cut it.

---

## 11. Alternate structures (and why I'd not lead with them)

| Shape | Strength | Failure mode |
|---|---|---|
| **Single-op descent** — follow one line of code through all 11 IRs | maximum concreteness, highly reproducible | narrow; the "why the substrate" argument gets buried in §9 of 10 |
| **Historical** — BLAS '79 → cuDNN → XLA → Triton → MLIR | strong narrative momentum, great characters | reads as a survey; dates fastest |
| **Taxonomy-first** — define arithmetic intensity, then classify every op | best as a reference; laser-clear | no arc; readers bounce at the table |
| **Recommended: hybrid** — profiler hook, descent as spine, ridge as hinge, dissolves as evidence, residue + numerics as honesty, inversion as close | the argument and the artifacts reinforce each other | needs discipline to keep §3 short |

---

## 12. Epigraph candidates

- "The purpose of computing is insight, not numbers." — Hamming (works against you; maybe as irony)
- Kung & Leiserson, 1978, on systolic arrays: computation and communication in the same regular structure. Nearly the thesis, 48 years early.
- Goto & van de Geijn, *Anatomy of a High-Performance Matrix Multiplication* (2008) — the canonical statement that GEMM performance is a memory-hierarchy problem wearing a math problem's clothes.

---

## 13. Reading list to cite (keep it to ~10, or it becomes a bibliography)

Lawson et al. 1979 (BLAS) · Kung & Leiserson 1978 (systolic arrays) · Goto & van de Geijn 2008 · Williams et al. 2009 (roofline) · Chetlur et al. 2014 (cuDNN, implicit GEMM) · Lattner et al. 2021 (MLIR) · Tillet et al. 2019 (Triton) · Dao et al. 2022 (FlashAttention) + FlashAttention-2/3 for the scheduling story · Micikevicius et al. 2018 (mixed precision) · Jouppi et al. 2017 (TPU) · Gu & Dao 2023 (Mamba, chunked scan).
