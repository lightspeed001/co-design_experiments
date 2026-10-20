CARGO ?= cargo
IVERILOG ?= iverilog
VVP ?= vvp
LLVM_PREFIX ?= /usr/lib/llvm-22
MLIR_OPT ?= $(LLVM_PREFIX)/bin/mlir-opt
BUILD_DIR := build
RTL_CPU := ../experiments/dotn-e2e/rtl/tiny_cpu_wait.sv
TB_CPU := tb/tb_toy8_cpu.sv
EXAMPLES := two_plus_three single_literal wrapping_add left_associative_sum

# mlir-sys 220 uses llvm-config; Melior 0.27 is pinned to the MLIR 22 C API.
# Leave MLIR_SYS_LINK_SHARED unset on systems that provide only static MLIR libraries.
export PATH := $(LLVM_PREFIX)/bin:$(HOME)/.cargo/bin:$(PATH)
export MLIR_SYS_220_PREFIX := $(LLVM_PREFIX)

.PHONY: all test format-check unit-tests compile-examples verify-mlir sim clean
all: test

test: format-check unit-tests verify-mlir sim

format-check:
	$(CARGO) fmt --all -- --check

unit-tests:
	$(CARGO) test --all-targets

compile-examples:
	mkdir -p $(BUILD_DIR)
	$(CARGO) run --quiet -- examples/two_plus_three.toy8 --out $(BUILD_DIR)/two_plus_three.hex --mlir $(BUILD_DIR)/two_plus_three.mlir --asm $(BUILD_DIR)/two_plus_three.asm
	$(CARGO) run --quiet -- examples/single_literal.toy8 --out $(BUILD_DIR)/single_literal.hex --mlir $(BUILD_DIR)/single_literal.mlir --asm $(BUILD_DIR)/single_literal.asm
	$(CARGO) run --quiet -- examples/wrapping_add.toy8 --out $(BUILD_DIR)/wrapping_add.hex --mlir $(BUILD_DIR)/wrapping_add.mlir --asm $(BUILD_DIR)/wrapping_add.asm
	$(CARGO) run --quiet -- examples/left_associative_sum.toy8 --out $(BUILD_DIR)/left_associative_sum.hex --mlir $(BUILD_DIR)/left_associative_sum.mlir --asm $(BUILD_DIR)/left_associative_sum.asm

verify-mlir: compile-examples
	@set -eu; for example in $(EXAMPLES); do \
		$(MLIR_OPT) $(BUILD_DIR)/$$example.mlir -o /dev/null; \
		echo "PASS: mlir-opt verified $$example.mlir"; \
	done

$(BUILD_DIR)/tb_toy8_cpu.vvp: $(TB_CPU) $(RTL_CPU)
	mkdir -p $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s tb_toy8_cpu -o $@ $(TB_CPU) $(RTL_CPU)

sim: compile-examples $(BUILD_DIR)/tb_toy8_cpu.vvp
	$(VVP) $(BUILD_DIR)/tb_toy8_cpu.vvp +ROM=$(BUILD_DIR)/two_plus_three.hex +EXPECTED=05 +WORDS=4
	$(VVP) $(BUILD_DIR)/tb_toy8_cpu.vvp +ROM=$(BUILD_DIR)/single_literal.hex +EXPECTED=A5 +WORDS=3
	$(VVP) $(BUILD_DIR)/tb_toy8_cpu.vvp +ROM=$(BUILD_DIR)/wrapping_add.hex +EXPECTED=00 +WORDS=4
	$(VVP) $(BUILD_DIR)/tb_toy8_cpu.vvp +ROM=$(BUILD_DIR)/left_associative_sum.hex +EXPECTED=64 +WORDS=6

clean:
	rm -rf $(BUILD_DIR) target
