// kernels.mlir — the input to `lower`.
//
// This is the *last* rung at which the model is still recognizable. `linalg.matmul`
// is a named op: it carries a promise about what it computes (a matrix product) in
// its name, so passes can reason about it without looking inside.
//
// We start here rather than at StableHLO because StableHLO is an out-of-tree dialect:
// upstream MLIR (which is what mlir-sys links) cannot parse or lower it without the
// StableHLO project's own libraries. See README.md — the driver's own "effective
// theory" has a cutoff, and this is it.

// --------------------------------------------------------------------------------------
// A dense GEMM in bf16 with an f32 accumulator. This is what every linear layer,
// every QKV projection, every expert, and both halves of attention compile to.
// --------------------------------------------------------------------------------------
func.func @matmul(%A: tensor<4096x4096xbf16>,
                  %B: tensor<4096x4096xbf16>) -> tensor<4096x4096xf32> {
  // An accumulator has to be born somewhere. `linalg.fill` is a named op too, and it
  // will dissolve into a loop nest one rung later, exactly like the matmul.
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4096x4096xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%init : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>

  // The subject of the post, in one line. Note what the op does NOT say: it does not
  // say how to tile, what order to reduce in, or where the data lives. Those are all
  // left to the lowering — which is why the lowering is where the performance is.
  %C = linalg.matmul ins(%A, %B : tensor<4096x4096xbf16>, tensor<4096x4096xbf16>)
                     outs(%acc : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %C : tensor<4096x4096xf32>
}

// --------------------------------------------------------------------------------------
// A fused elementwise op — the residue from section 6, for comparison. It shares the
// loop-nest treatment with the matmul but has *no* reduction, so it has no
// associativity to exploit and no data reuse to tile for. It is pure traffic.
// --------------------------------------------------------------------------------------
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @silu_mul(%gate: tensor<8x14336xf32>,
                    %up: tensor<8x14336xf32>) -> tensor<8x14336xf32> {
  %out = tensor.empty() : tensor<8x14336xf32>
  %r = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%gate, %up : tensor<8x14336xf32>, tensor<8x14336xf32>)
      outs(%out : tensor<8x14336xf32>) {
    // SiLU(gate) * up, the SwiGLU sandwich. One exp, one divide, one multiply.
    ^bb0(%g: f32, %u: f32, %o: f32):
      %neg = arith.negf %g : f32
      %e = math.exp %neg : f32
      %one = arith.constant 1.0 : f32
      %den = arith.addf %one, %e : f32
      %s = arith.divf %g, %den : f32
      %y = arith.mulf %s, %u : f32
      linalg.yield %y : f32
  } -> tensor<8x14336xf32>
  return %r : tensor<8x14336xf32>
}
