// RUNG 6 — `convert-vector-to-gpu` with nvgpu enabled. CASUALTY: the shape of a matmul.
//
// The contraction has been quantized to the register layout of one warp-level
// instruction. A 16x16x16 contraction no longer exists as a concept; what exists is
// `m16n8k16`, and the operands are not matrices but *fragments*: 4 registers for A,
// 2 for B, 4 fp32 accumulators for C/D, distributed across 32 lanes.
//
// (Attribute spelling for mmaShape has moved around between LLVM releases; treat the
// exact literal as version-dependent. That fragility is itself evidence for section 3.)
%0 = nvgpu.mma.sync (%a_frag, %b_frag, %c_frag) {mmaShape = [16, 8, 16]}
       : (vector<16x16xbf16>, vector<16x8xbf16>, vector<16x8xf32>) -> vector<16x8xf32>
