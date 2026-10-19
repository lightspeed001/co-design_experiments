// RUNG 3 — `convert-linalg-to-loops`. CASUALTY: the structured op itself.
//
// The affine maps are gone. What is left is a traversal. This is the most important
// forty lines in the whole descent, because it is where performance is decided and
// where the algebra stops being visible:
//
//   per inner iteration: 2 loads (2B each) + 1 load (4B) + 1 store (4B) = 12 bytes
//                        for 2 FLOPs
//   arithmetic intensity as written: 2/12 = 0.17 FLOP/byte
//   ridge point:                      295 FLOP/byte
//
// i.e. this loop nest asks the memory system for ~1,700x more bytes per FLOP than the
// machine can afford to pay for. No amount of instruction scheduling fixes that. Only
// the loop ORDER does — which is why every optimization from here up is a
// reassociation, and why the topology below this line is chosen by the cache.
func.func @matmul(%arg0: memref<4096x4096xbf16>,
                  %arg1: memref<4096x4096xbf16>,
                  %arg2: memref<4096x4096xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4096 = arith.constant 4096 : index
  scf.for %i = %c0 to %c4096 step %c1 {
    scf.for %j = %c0 to %c4096 step %c1 {
      scf.for %k = %c0 to %c4096 step %c1 {          // <- the reduction, now an iv
        %a   = memref.load %arg0[%i, %k] : memref<4096x4096xbf16>
        %b   = memref.load %arg1[%k, %j] : memref<4096x4096xbf16>
        %acc = memref.load %arg2[%i, %j] : memref<4096x4096xf32>
        %ae  = arith.extf %a : bf16 to f32
        %be  = arith.extf %b : bf16 to f32
        %p   = arith.mulf %ae, %be : f32
        %s   = arith.addf %acc, %p : f32
        memref.store %s, %arg2[%i, %j] : memref<4096x4096xf32>
      }
    }
  }
  return
}
