// RUNG 5 — after scheduling: tile, then vectorize. CASUALTY: the scalar loop nest.
//
// This rung is not reachable from plain pass flags alone; it requires a *schedule*
// (transform dialect or a hand-written pass). That is the point: the algebra is
// unchanged, but the nesting order is now an explicit, load-bearing object in the IR.
//
// The outer loop is `scf.forall` (all iterations independent — parallel), and inside
// it the reduction is handed to `vector.contract`, whose attribute set is the SAME
// triangle of index maps we saw at rung 1. The tower rhymes.
/* %C = */ scf.forall (%i, %j) in (256, 256) shared_outs(%o = %C) -> (tensor<4096x4096xf32>) {
  %a = vector.transfer_read %A[%i, %k], %z : tensor<4096x4096xbf16>, vector<16x16xbf16>
  %b = vector.transfer_read %B[%k, %j], %z : tensor<4096x4096xbf16>, vector<16x16xbf16>
  %c = vector.transfer_read %o[%i, %j], %z : tensor<4096x4096xf32>, vector<16x16xf32>

  // One op. Three maps. Two parallels and a reduction. Same shape as rung 1 — but now
  // `vector.contract` is a *hardware-shaped* object: it has a fixed size, and the
  // vectorizer chose that size because the silicon implements it.
  %d = vector.contract {
         indexing_maps = [#map_a, #map_b, #map_c],
         iterator_types = ["parallel", "parallel", "reduction"],
         kind = #vector.kind<add>
       } %a, %b, %c : vector<16x16xbf16>, vector<16x16xbf16> into vector<16x16xf32>

  vector.transfer_write %d, %o[%i, %j] : vector<16x16xf32>, tensor<4096x4096xf32>
}
