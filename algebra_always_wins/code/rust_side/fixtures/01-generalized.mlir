// RUNG 1 — `linalg-generalize-named-ops`. CASUALTY: the word "matmul".
//
// Nothing about the arithmetic changed. What changed is that the compiler no longer
// has an op it can special-case; it has an index space, a distribution of those
// indices into parallel/reduction, and a scalar body. From here down, EVERY op in
// the program looks like this, which is precisely why one set of transforms can
// optimize all of them.
#map  = affine_map<(d0, d1, d2) -> (d0, d2)>   // A[i, k]
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>   // B[k, j]
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>   // C[i, j]

func.func @matmul(%arg0: tensor<4096x4096xbf16>,
                  %arg1: tensor<4096x4096xbf16>) -> tensor<4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4096x4096xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>

  // Read the body. This is the whole of a matrix multiply, and it is four lines of
  // scalar arithmetic — no reduction hardware, no tensor semantics, no "dot". The
  // reduction lives in `iterator_types`, not in the operations.
  %2 = linalg.generic {
         indexing_maps = [#map, #map1, #map2],
         iterator_types = ["parallel", "parallel", "reduction"]
       } ins(%arg0, %arg1 : tensor<4096x4096xbf16>, tensor<4096x4096xbf16>)
         outs(%1 : tensor<4096x4096xf32>) {
       ^bb0(%in: bf16, %in_0: bf16, %out: f32):
         %3 = arith.extf %in : bf16 to f32      // widen — precision decision, rung 1
         %4 = arith.extf %in_0 : bf16 to f32    // widen
         %5 = arith.mulf %3, %4 : f32
         %6 = arith.addf %out, %5 : f32         // the associative combine
         linalg.yield %6 : f32
       } -> tensor<4096x4096xf32>
  return %2 : tensor<4096x4096xf32>
}
