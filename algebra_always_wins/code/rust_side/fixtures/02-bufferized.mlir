// RUNG 2 — `one-shot-bufferize`. CASUALTY: value semantics.
//
// Every tensor is now a memref: base pointer, sizes, strides, offset. Layout is no
// longer a detail that the compiler may defer — it is in the type, and every access
// is an affine function of it. This is also the rung where the output buffer gets an
// allocation, i.e. where the program acquires a working set and therefore a
// relationship with the memory hierarchy.
func.func @matmul(%arg0: memref<4096x4096xbf16>,
                  %arg1: memref<4096x4096xbf16>) -> memref<4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<4096x4096xf32>
  linalg.fill ins(%cst : f32) outs(%0 : memref<4096x4096xf32>)
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : memref<4096x4096xbf16>, memref<4096x4096xbf16>)
    outs(%0 : memref<4096x4096xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %3 = arith.extf %in : bf16 to f32
    %4 = arith.extf %in_0 : bf16 to f32
    %5 = arith.mulf %3, %4 : f32
    %6 = arith.addf %out, %5 : f32
    linalg.yield %6 : f32
  }
  return %0 : memref<4096x4096xf32>
}
