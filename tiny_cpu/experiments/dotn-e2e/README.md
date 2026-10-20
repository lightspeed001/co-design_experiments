# Tiny8 `dotN` compiler-to-RTL path and regression

This project connects a small source-to-Tiny8 compiler slice to the CPU-visible descriptor block, DMA, round-robin single-port RAM arbiter, and unsigned ready/valid `dotN` MAC. It also keeps the hand-assembled CPU smoke test and the bus-level descriptor regression.

## Run it

Requirements: Python 3 and Icarus Verilog (`iverilog`/`vvp`) with SystemVerilog-2012 support.

```sh
make test
```

`make test` runs the source-compiler unit checks, the hand-assembled CPU-instruction smoke test, the compiler-generated CPU program, and the multi-case MMIO/DMA regression. To run one target separately:

```sh
make compiler-check  # source/type/layout validation tests
make compile         # emit generated.asm and generated.hex from dotn_call.toy8
make sim             # CPU executes hand-assembled program.asm
make sim-compiler    # run compiler-generated ROM on the same CPU/system RTL
make regression      # MMIO/DMA/arbiter/core boundary and operand regression
```

## Compiler-generated execution

`dotn_call.toy8` is a deliberately restricted source language with typed `u8` array literals, one `dot_u8_into` call, and one expected-result check. `dotn_source_compiler.py` checks the array sizes, length, fixed RAM layout, result width, and expected bytes, then emits `generated.asm`. The existing `assemble.py` translates that assembly into `generated.hex`; `tb_dotn_e2e.sv` loads that ROM with `+ROM=generated.hex` and runs the Tiny8 CPU through the normal MMIO/DMA/RAM/MAC path.

This is a **minimal Python compiler prototype**, separate from the implemented Rust/Melior scalar Toy8 v1 compiler in [Foundations 2](../../part-2-toy-compiler.md). It demonstrates one target-specific `dot_u8_into` lowering and execution path, but does not integrate MLIR, implement the Foundations 2 expression language, or provide a full compiler/runtime ABI. The corresponding write-up is [Experiments Step 4](../../experiments-3-dotn-compiler-generated.md).

## CPU-instruction smoke test

`program.asm` is the 50-instruction hand-assembled version of the `LEN=4` test. It writes `A=[1,2,3,4]` at `0x20`, `B=[5,6,7,8]` at `0x30`, submits the descriptor with destination `0x40`, polls `DONE`, checks `ERROR` and the result bytes, and writes a pass/fail marker to `0x50`. Its polling loop reads scratch byte `0x60`, intentionally creating CPU/DMA RAM contention so `dmem_ready` wait behavior is exercised.

## Regression cases

`tb/tb_dotn_regression.sv` drives the CPU-side bus directly to cover:

- Valid `LEN=0`, `LEN=1`, `LEN=4`, and `LEN=16` transactions. The maximum-length case uses all-`0xFF` operands and checks the full 20-bit maximum result.
- A valid exclusive-end boundary at `0xF0` and source/destination overlap, including a result range that overwrites an input byte only after all input reads finish.
- Five rejection cases: `LEN=17`, A-range overflow, B-range overflow, destination-range overflow, and an 8-bit-wrapping source base. Each must set `DONE`/`ERROR`, issue no DMA RAM or stream transfers, and preserve the in-range destination sentinel bytes.
- Sixteen deterministic pseudo-random operand vectors, one for each nonzero supported length.
- For every valid job, exactly `2 × LEN` DMA source reads, `LEN` stream handshakes, and three result writes.

## Scope and limitations

The Tiny8 CPU RTL implements the nine-instruction ISA with a `dmem_ready` wait path. The compiler prototype generates one fixed-layout accelerator call; the CPU smoke test and compiler-generated test are directed cases. The bus-level regression is not a CPU-instruction test. The RAM model has a combinational read port and is intended for teaching.

These are directed RTL simulations, not synthesis, FPGA implementation, physical design, or a software performance comparison. Reported cycle counts apply only to this exact testbench and RTL schedule; they are not measured hardware latency.
