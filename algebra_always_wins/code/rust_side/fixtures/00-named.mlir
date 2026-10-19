// RUNG 0 — as written. The op still has a name, and the name still means something.
func.func @matmul(%A: tensor<4096x4096xbf16>,
                  %B: tensor<4096x4096xbf16>) -> tensor<4096x4096xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4096x4096xf32>
  %acc  = linalg.fill ins(%zero : f32) outs(%init : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  %C = linalg.matmul ins(%A, %B : tensor<4096x4096xbf16>, tensor<4096x4096xbf16>)
                     outs(%acc : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %C : tensor<4096x4096xf32>
}
