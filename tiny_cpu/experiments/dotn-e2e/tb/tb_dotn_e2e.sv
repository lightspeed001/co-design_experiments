`timescale 1ns/1ps

// Self-checking integration test: Tiny8 instructions program a real descriptor,
// DMA reads shared RAM, dotN consumes four pairs, and three result bytes return to RAM.
module tb_dotn_e2e;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic [7:0]  imem_addr;
    logic [15:0] imem_rdata;
    logic        dmem_req;
    logic        dmem_we;
    logic [7:0]  dmem_addr;
    logic [7:0]  dmem_wdata;
    logic        dmem_ready;
    logic [7:0]  dmem_rdata;
    logic [7:0]  debug_pc;
    logic [7:0]  debug_acc;
    logic        halted;
    logic        done;
    logic        error;

    logic [15:0] instruction_mem [0:255];
    string rom_file;
    integer i;

    assign imem_rdata = instruction_mem[imem_addr];

    tiny_cpu_wait u_cpu (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr), .imem_rdata(imem_rdata),
        .dmem_req(dmem_req), .dmem_we(dmem_we), .dmem_addr(dmem_addr),
        .dmem_wdata(dmem_wdata), .dmem_ready(dmem_ready), .dmem_rdata(dmem_rdata),
        .debug_pc(debug_pc), .debug_acc(debug_acc), .halted(halted)
    );

    dotn_system u_system (
        .clk(clk), .rst_n(rst_n),
        .cpu_req(dmem_req), .cpu_we(dmem_we), .cpu_addr(dmem_addr),
        .cpu_wdata(dmem_wdata), .cpu_ready(dmem_ready), .cpu_rdata(dmem_rdata),
        .done(done), .error(error)
    );

    integer cycle_count = 0;
    integer start_cycle = -1;
    integer done_cycle = -1;
    integer dma_reads = 0;
    integer dma_writes = 0;
    integer stream_handshakes = 0;
    integer cpu_wait_cycles = 0;
    logic previous_done = 1'b0;

    // Count real edge handshakes, not state estimates or calculated schedules.
    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        if (rst_n) begin
            if (dmem_req && dmem_we && dmem_ready && dmem_addr == 8'hF0)
                start_cycle = cycle_count;
            if (u_system.dma_req && u_system.dma_ready) begin
                if (u_system.dma_we) dma_writes = dma_writes + 1;
                else                 dma_reads  = dma_reads + 1;
            end
            if (u_system.s_valid && u_system.s_ready)
                stream_handshakes = stream_handshakes + 1;
            if (u_system.cpu_ram_req && !u_system.cpu_ram_ready && u_system.dma_req)
                cpu_wait_cycles = cpu_wait_cycles + 1;
        end
        #1; // Observe nonblocking state updates from the rising edge.
        if (rst_n && done && !previous_done && done_cycle < 0)
            done_cycle = cycle_count;
        previous_done = done;
    end

    initial begin
        // Default unused instruction words to HALT, then load the selected assembled image.
        for (i = 0; i < 256; i = i + 1)
            instruction_mem[i] = 16'hF000;
        rom_file = "program.hex";
        if (!$value$plusargs("ROM=%s", rom_file))
            rom_file = "program.hex";
        $display("ROM: %s", rom_file);
        $readmemh(rom_file, instruction_mem);

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        wait (halted === 1'b1);
        @(negedge clk);

        // The CPU program also checks these bytes and writes the PASS marker.
        if (u_system.u_arbiter.ram[64] !== 8'h46 ||
            u_system.u_arbiter.ram[65] !== 8'h00 ||
            u_system.u_arbiter.ram[66] !== 8'h00) begin
            $error("bad destination: %02h %02h %02h",
                   u_system.u_arbiter.ram[64],
                   u_system.u_arbiter.ram[65],
                   u_system.u_arbiter.ram[66]);
            $fatal(1, "end-to-end result mismatch");
        end
        if (u_system.u_arbiter.ram[80] !== 8'hA5)
            $fatal(1, "CPU program did not write the PASS marker (got %02h)",
                   u_system.u_arbiter.ram[80]);
        if (done !== 1'b1 || error !== 1'b0)
            $fatal(1, "status mismatch: DONE=%b ERROR=%b", done, error);
        if (dma_reads != 8 || dma_writes != 3 || stream_handshakes != 4)
            $fatal(1, "transaction counts mismatch: reads=%0d writes=%0d stream=%0d",
                   dma_reads, dma_writes, stream_handshakes);
        if (cpu_wait_cycles == 0)
            $fatal(1, "expected at least one CPU wait during DMA/RAM contention");
        if (start_cycle < 0 || done_cycle < start_cycle)
            $fatal(1, "missing START/DONE cycle observations");

        $display("PASS: dotN([1,2,3,4],[5,6,7,8]) = 70 (0x00046)");
        $display("PASS: destination bytes = %02h %02h %02h; CPU marker = %02h",
                 u_system.u_arbiter.ram[64],
                 u_system.u_arbiter.ram[65],
                 u_system.u_arbiter.ram[66],
                 u_system.u_arbiter.ram[80]);
        $display("SIM: start-to-DONE=%0d clock edges; DMA reads=%0d, writes=%0d, stream handshakes=%0d, CPU wait edges=%0d",
                 done_cycle - start_cycle, dma_reads, dma_writes,
                 stream_handshakes, cpu_wait_cycles);
        $finish;
    end

    initial begin
        // A bounded timeout makes hangs and broken polling fail instead of stalling the run.
        #200000;
        $fatal(1, "timeout: PC=%02h ACC=%02h DONE=%b", debug_pc, debug_acc, done);
    end
endmodule
