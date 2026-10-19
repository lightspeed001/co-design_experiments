// RUNG 4 — conversion to the LLVM dialect. CASUALTY: MLIR itself.
//
// Excerpt, heavily abridged: the real output is ~10x this, because a memref descriptor
// is three fields and every `memref.load` becomes a getelementptr + a load. Note what
// is NOT here: no shape, no iterator types, no index maps. Only addresses, branches,
// and a multiply-add whose order is now fixed by the register allocator.
//
// Key detail: `%s0` and `%s1` are the strides, extracted from the descriptor at
// runtime. The *layout* — the thing that decides whether this kernel is 3% or 60% of
// peak — is now an integer that the CPU reads out of memory.
llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  ...
^bb5:  // %k loop header
  %s0 = llvm.extractvalue %desc_a[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %off = llvm.getelementptr %base[%i, %s0, %k] : ...
  %a    = llvm.load %off : !llvm.ptr -> bf16
  %ae   = llvm.fpext %a : bf16 to f32
  %p    = llvm.fmul %ae, %be : f32
  %s    = llvm.fadd %acc, %p : f32
  llvm.store %s, %out : !llvm.ptr
  llvm.cond_br %cond, ^bb5, ^bb6
}
