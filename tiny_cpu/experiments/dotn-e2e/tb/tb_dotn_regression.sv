`timescale 1ns/1ps

// Directed plus deterministic-random regression for the MMIO/DMA/arbiter/MAC path.
// The step-2 CPU-instruction integration test remains in tb_dotn_e2e.sv; this
// bench drives the same CPU-side data bus directly so many descriptors fit quickly.
module tb_dotn_regression;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cpu_req = 1'b0;
    logic        cpu_we = 1'b0;
    logic [7:0]  cpu_addr = 8'h00;
    logic [7:0]  cpu_wdata = 8'h00;
    logic        cpu_ready;
    logic [7:0]  cpu_rdata;
    logic        done;
    logic        error;

    dotn_system u_system (
        .clk(clk), .rst_n(rst_n),
        .cpu_req(cpu_req), .cpu_we(cpu_we), .cpu_addr(cpu_addr),
        .cpu_wdata(cpu_wdata), .cpu_ready(cpu_ready), .cpu_rdata(cpu_rdata),
        .done(done), .error(error)
    );

    integer dma_reads = 0;
    integer dma_writes = 0;
    integer stream_handshakes = 0;
    logic [7:0] a_values [0:15];
    logic [7:0] b_values [0:15];
    logic [31:0] rng_state = 32'h5EED_C0DE;

    // Count only accepted RAM and stream transactions at rising clock edges.
    always @(posedge clk) begin
        if (rst_n) begin
            if (u_system.dma_req && u_system.dma_ready) begin
                if (u_system.dma_we) dma_writes = dma_writes + 1;
                else                 dma_reads  = dma_reads + 1;
            end
            if (u_system.s_valid && u_system.s_ready)
                stream_handshakes = stream_handshakes + 1;
        end
    end

    function automatic [31:0] lcg_next(input [31:0] value);
        // Fixed-seed linear congruential sequence keeps randomized runs repeatable.
        lcg_next = value * 32'd1664525 + 32'd1013904223;
    endfunction

    task automatic cpu_write(input logic [7:0] address, input logic [7:0] data);
        integer granted;
        begin
            // Hold the CPU-side request stable until RAM grants it or MMIO completes it.
            @(negedge clk);
            cpu_req   = 1'b1;
            cpu_we    = 1'b1;
            cpu_addr  = address;
            cpu_wdata = data;
            granted = 0;
            while (!granted) begin
                @(posedge clk);
                if (cpu_ready) granted = 1;
            end
            @(negedge clk);
            cpu_req   = 1'b0;
            cpu_we    = 1'b0;
            cpu_addr  = 8'h00;
            cpu_wdata = 8'h00;
        end
    endtask

    task automatic cpu_read(input logic [7:0] address, output logic [7:0] data);
        integer granted;
        begin
            // Capture read data only on the edge where the CPU-facing request is ready.
            @(negedge clk);
            cpu_req   = 1'b1;
            cpu_we    = 1'b0;
            cpu_addr  = address;
            cpu_wdata = 8'h00;
            granted = 0;
            while (!granted) begin
                @(posedge clk);
                if (cpu_ready) begin
                    data = cpu_rdata;
                    granted = 1;
                end
            end
            @(negedge clk);
            cpu_req   = 1'b0;
            cpu_we    = 1'b0;
            cpu_addr  = 8'h00;
        end
    endtask

    task automatic wait_for_done;
        integer waited;
        begin
            waited = 0;
            while ((done !== 1'b1) && waited < 500) begin
                @(posedge clk);
                waited = waited + 1;
            end
            if (done !== 1'b1)
                $fatal(1, "timeout waiting for DONE");
            @(negedge clk);
        end
    endtask

    task automatic clear_transaction_counts;
        begin
            @(negedge clk);
            dma_reads = 0;
            dma_writes = 0;
            stream_handshakes = 0;
        end
    endtask

    task automatic run_valid_case(
        input integer case_id,
        input integer len,
        input integer a_base,
        input integer b_base,
        input integer dst_base,
        input integer pattern
    );
        integer k;
        logic [7:0] a_byte;
        logic [7:0] b_byte;
        logic [7:0] got0;
        logic [7:0] got1;
        logic [7:0] got2;
        logic [7:0] status;
        logic [15:0] product;
        logic [19:0] expected;
        begin
            expected = 20'd0;
            for (k = 0; k < len; k = k + 1) begin
                if (pattern == 0) begin
                    a_byte = k + 1;
                    b_byte = k + 5;
                end else if (pattern == 1) begin
                    a_byte = 8'hFF;
                    b_byte = 8'hFF;
                end else begin
                    rng_state = lcg_next(rng_state);
                    a_byte = rng_state[7:0];
                    rng_state = lcg_next(rng_state);
                    b_byte = rng_state[7:0];
                end
                a_values[k] = a_byte;
                b_values[k] = b_byte;
                // Extend both bytes before multiply so the reference product is 16 bits.
                product = {8'h00, a_byte} * {8'h00, b_byte};
                expected = expected + {4'b0000, product};
            end

            // Load source bytes through the CPU-facing RAM port before START.
            for (k = 0; k < len; k = k + 1) begin
                cpu_write(a_base + k, a_values[k]);
                cpu_write(b_base + k, b_values[k]);
            end

            clear_transaction_counts();
            cpu_write(8'hF2, len[7:0]);
            cpu_write(8'hF7, a_base[7:0]);
            cpu_write(8'hF8, b_base[7:0]);
            cpu_write(8'hF9, dst_base[7:0]);
            cpu_write(8'hF0, 8'h01);

            if (done !== 1'b0 || error !== 1'b0)
                $fatal(1, "case %0d: accepted START did not clear status", case_id);
            wait_for_done();

            cpu_read(8'hF1, status);
            if (status !== 8'h01)
                $fatal(1, "case %0d: DONE register was %02h", case_id, status);
            cpu_read(8'hF3, status);
            if (status !== 8'h00)
                $fatal(1, "case %0d: unexpected ERROR register %02h", case_id, status);

            // Read the written bytes through the CPU bus, including overlap cases.
            cpu_read(dst_base, got0);
            cpu_read(dst_base + 1, got1);
            cpu_read(dst_base + 2, got2);
            if (got0 !== expected[7:0] || got1 !== expected[15:8] ||
                got2 !== {4'b0000, expected[19:16]})
                $fatal(1, "case %0d: expected %02h %02h %01h, got %02h %02h %02h",
                       case_id, expected[7:0], expected[15:8], expected[19:16],
                       got0, got1, got2);
            if (dma_reads != 2 * len || dma_writes != 3 ||
                stream_handshakes != len)
                $fatal(1, "case %0d: reads=%0d writes=%0d handshakes=%0d for LEN=%0d",
                       case_id, dma_reads, dma_writes, stream_handshakes, len);

            $display("PASS: valid case %0d LEN=%0d result=%05h reads=%0d writes=%0d streams=%0d",
                     case_id, len, expected, dma_reads, dma_writes, stream_handshakes);
        end
    endtask

    task automatic run_invalid_case(
        input integer case_id,
        input integer len,
        input integer a_base,
        input integer b_base,
        input integer dst_base,
        input integer sentinel_base,
        input integer sentinel_len
    );
        integer k;
        logic [7:0] status;
        logic [7:0] sentinel;
        begin
            // Seed the in-range portion of the destination with a value that must survive rejection.
            for (k = 0; k < sentinel_len; k = k + 1)
                cpu_write(sentinel_base + k, 8'hD0 + k);

            clear_transaction_counts();
            cpu_write(8'hF2, len[7:0]);
            cpu_write(8'hF7, a_base[7:0]);
            cpu_write(8'hF8, b_base[7:0]);
            cpu_write(8'hF9, dst_base[7:0]);
            cpu_write(8'hF0, 8'h01);

            if (done !== 1'b1 || error !== 1'b1)
                $fatal(1, "invalid case %0d was not rejected with DONE/ERROR", case_id);
            cpu_read(8'hF1, status);
            if (status !== 8'h01)
                $fatal(1, "invalid case %0d: DONE register was %02h", case_id, status);
            cpu_read(8'hF3, status);
            if (status !== 8'h01)
                $fatal(1, "invalid case %0d: ERROR register was %02h", case_id, status);
            if (dma_reads != 0 || dma_writes != 0 || stream_handshakes != 0)
                $fatal(1, "invalid case %0d transferred data: reads=%0d writes=%0d streams=%0d",
                       case_id, dma_reads, dma_writes, stream_handshakes);

            for (k = 0; k < sentinel_len; k = k + 1) begin
                cpu_read(sentinel_base + k, sentinel);
                if (sentinel !== (8'hD0 + k))
                    $fatal(1, "invalid case %0d changed sentinel at %02h", case_id, sentinel_base + k);
            end
            $display("PASS: invalid case %0d rejected without DMA/stream transfers", case_id);
        end
    endtask

    integer trial;

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        // Empty, single-term, fixed example, maximum sum, and overlap/end-boundary cases.
        run_valid_case(1, 0, 8'h20, 8'h30, 8'h80, 0);
        run_valid_case(2, 1, 8'h20, 8'h30, 8'h80, 1);
        run_valid_case(3, 16, 8'h20, 8'h40, 8'h60, 1);
        run_valid_case(4, 4, 8'h20, 8'h30, 8'h80, 0);
        run_valid_case(5, 4, 8'h20, 8'h30, 8'h22, 0); // DST overlaps A; all sources are read first.
        run_valid_case(6, 1, 8'hEF, 8'hE0, 8'hED, 1); // Exact exclusive-end 0xF0; output overlaps A[0].

        // Fixed-seed random vectors exercise every nonzero supported length once.
        for (trial = 1; trial <= 16; trial = trial + 1)
            run_valid_case(10 + trial, trial, 8'h20, 8'h50, 8'h80, 2);

        // Invalid length, each input's exclusive-end check, output end, and 8-bit wraparound.
        run_invalid_case(30, 17, 8'h20, 8'h30, 8'h80, 8'h80, 3);
        run_invalid_case(31, 2, 8'hEF, 8'h30, 8'h80, 8'h80, 3);
        run_invalid_case(32, 2, 8'h20, 8'hEF, 8'h80, 8'h80, 3);
        run_invalid_case(33, 1, 8'h20, 8'h30, 8'hEE, 8'hEE, 2);
        run_invalid_case(34, 1, 8'hFF, 8'h30, 8'h80, 8'h80, 3);

        $display("PASS: valid cases=22 (6 directed + 16 fixed-seed), invalid cases=5; random seed=5EEDC0DE");
        $display("PASS: all directed, boundary, overlap, invalid, and 16 deterministic-random cases");
        $finish;
    end

    initial begin
        // Catch deadlocks in any descriptor or handshaking scenario.
        #2000000;
        $fatal(1, "regression timeout");
    end
endmodule
