IVERILOG ?= iverilog
VVP ?= vvp
PYTHON ?= python3
BUILD_DIR := build
RTL := rtl/tiny_cpu_wait.sv rtl/dotn_mac_core.sv rtl/ram_arbiter.sv rtl/dotn_dma.sv rtl/dotn_system.sv
TB_E2E := tb/tb_dotn_e2e.sv
TB_REGRESSION := tb/tb_dotn_regression.sv

.PHONY: all assemble compile compiler-check sim sim-compiler regression test clean
all: test

compiler-check:
	$(PYTHON) -m unittest -v test_dotn_source_compiler.py

assemble:
	$(PYTHON) assemble.py program.asm program.hex

compile:
	$(PYTHON) dotn_source_compiler.py dotn_call.toy8 generated.asm
	$(PYTHON) assemble.py generated.asm generated.hex

sim: assemble
	mkdir -p $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s tb_dotn_e2e -o $(BUILD_DIR)/tb_dotn_e2e.vvp $(RTL) $(TB_E2E)
	$(VVP) $(BUILD_DIR)/tb_dotn_e2e.vvp

sim-compiler: compile
	mkdir -p $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s tb_dotn_e2e -o $(BUILD_DIR)/tb_dotn_compiler.vvp $(RTL) $(TB_E2E)
	$(VVP) $(BUILD_DIR)/tb_dotn_compiler.vvp +ROM=generated.hex

regression:
	mkdir -p $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s tb_dotn_regression -o $(BUILD_DIR)/tb_dotn_regression.vvp $(RTL) $(TB_REGRESSION)
	$(VVP) $(BUILD_DIR)/tb_dotn_regression.vvp

test: compiler-check sim sim-compiler regression

clean:
	rm -rf $(BUILD_DIR)
