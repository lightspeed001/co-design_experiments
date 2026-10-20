`timescale 1ns/1ps

// End-to-end test for Rust/Melior-generated ROM images on the Tiny8 CPU RTL.
module tb_toy8_cpu;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic [7:0] imem_addr;
    logic [15:0] imem_rdata;
    logic dmem_req;
    logic dmem_we;
    logic [7:0] dmem_addr;
    logic [7:0] dmem_wdata;
    logic dmem_ready;
    logic [7:0] dmem_rdata;
    logic [7:0] debug_pc;
    logic [7:0] debug_acc;
    logic halted;

    logic [15:0] instruction_mem [0:255];
    logic [7:0] data_mem [0:255];
    string rom_file;
    integer expected_arg;
    integer words_arg;
    integer i;

    assign imem_rdata = instruction_mem[imem_addr];
    assign dmem_ready = 1'b1;
    assign dmem_rdata = data_mem[dmem_addr];

    tiny_cpu_wait cpu (
        .clk(clk),
        .rst_n(rst_n),
        .imem_addr(imem_addr),
        .imem_rdata(imem_rdata),
        .dmem_req(dmem_req),
        .dmem_we(dmem_we),
        .dmem_addr(dmem_addr),
        .dmem_wdata(dmem_wdata),
        .dmem_ready(dmem_ready),
        .dmem_rdata(dmem_rdata),
        .debug_pc(debug_pc),
        .debug_acc(debug_acc),
        .halted(halted)
    );

    always_ff @(posedge clk) begin
        if (rst_n && dmem_req && dmem_we && dmem_ready)
            data_mem[dmem_addr] <= dmem_wdata;
    end

    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            instruction_mem[i] = 16'hF000;
            data_mem[i] = 8'h00;
        end

        rom_file = "build/two_plus_three.hex";
        if (!$value$plusargs("ROM=%s", rom_file))
            rom_file = "build/two_plus_three.hex";
        if (!$value$plusargs("EXPECTED=%h", expected_arg))
            expected_arg = 8'h05;
        if (!$value$plusargs("WORDS=%d", words_arg))
            words_arg = 4;

        $display("ROM: %s; expected result=0x%02h; program words=%0d",
                 rom_file, expected_arg[7:0], words_arg);
        $readmemh(rom_file, instruction_mem);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        wait (halted === 1'b1);
        @(negedge clk);

        if (data_mem[8'h10] !== expected_arg[7:0])
            $fatal(1, "MEM[0x10]=0x%02h; expected 0x%02h",
                   data_mem[8'h10], expected_arg[7:0]);
        if (debug_acc !== expected_arg[7:0])
            $fatal(1, "A=0x%02h at HALT; expected 0x%02h",
                   debug_acc, expected_arg[7:0]);
        if (debug_pc !== (words_arg - 1))
            $fatal(1, "HALT reached at PC=0x%02h; expected 0x%02h",
                   debug_pc, words_arg - 1);

        $display("PASS: Tiny8 main returned 0x%02h to MEM[0x10]; PC=0x%02h",
                 data_mem[8'h10], debug_pc);
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "timeout: PC=0x%02h A=0x%02h halted=%b",
               debug_pc, debug_acc, halted);
    end
endmodule
