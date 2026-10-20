`timescale 1ns/1ps

// Serial DMA for one dotN job on a single-port RAM: read A, read B, send one
// stream pair, wait for core completion, then write three little-endian bytes.
module dotn_dma (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        job_start,
    input  logic [7:0]  job_len,
    input  logic [7:0]  job_a_base,
    input  logic [7:0]  job_b_base,
    input  logic [7:0]  job_dst_base,

    output logic        dma_req,
    output logic        dma_we,
    output logic [7:0]  dma_addr,
    output logic [7:0]  dma_wdata,
    input  logic        dma_ready,
    input  logic [7:0]  dma_rdata,

    output logic        core_job_start,
    output logic [4:0]  core_job_len,
    output logic        s_valid,
    input  logic        s_ready,
    output logic [7:0]  s_a,
    output logic [7:0]  s_b,
    input  logic        core_done,
    input  logic [19:0] core_result,

    output logic        job_done,
    output logic        busy,
    output logic [19:0] result
);
    typedef enum logic [3:0] {
        DMA_IDLE,
        DMA_START_CORE,
        DMA_READ_A,
        DMA_READ_B,
        DMA_SEND_PAIR,
        DMA_WAIT_CORE,
        DMA_WRITE_R0,
        DMA_WRITE_R1,
        DMA_WRITE_R2
    } state_t;
    state_t state;

    logic [7:0] len_latched;
    logic [7:0] a_base_latched;
    logic [7:0] b_base_latched;
    logic [7:0] dst_base_latched;
    logic [4:0] pair_index;
    logic [7:0] a_hold;
    logic [7:0] b_hold;
    logic [19:0] result_hold;

    assign busy          = (state != DMA_IDLE);
    assign core_job_start = (state == DMA_START_CORE);
    assign core_job_len   = len_latched[4:0];
    assign s_a            = a_hold;
    assign s_b            = b_hold;
    assign result         = result_hold;

    // The result-write grant and job completion coincide on the final write edge.
    always @* begin
        dma_req   = 1'b0;
        dma_we    = 1'b0;
        dma_addr  = 8'h00;
        dma_wdata = 8'h00;
        s_valid   = 1'b0;
        job_done  = 1'b0;

        case (state)
            DMA_READ_A: begin
                dma_req  = 1'b1;
                dma_addr = a_base_latched + {3'b000, pair_index};
            end
            DMA_READ_B: begin
                dma_req  = 1'b1;
                dma_addr = b_base_latched + {3'b000, pair_index};
            end
            DMA_SEND_PAIR: s_valid = 1'b1;
            DMA_WRITE_R0: begin
                dma_req   = 1'b1;
                dma_we    = 1'b1;
                dma_addr  = dst_base_latched;
                dma_wdata = result_hold[7:0];
            end
            DMA_WRITE_R1: begin
                dma_req   = 1'b1;
                dma_we    = 1'b1;
                dma_addr  = dst_base_latched + 8'd1;
                dma_wdata = result_hold[15:8];
            end
            DMA_WRITE_R2: begin
                dma_req   = 1'b1;
                dma_we    = 1'b1;
                dma_addr  = dst_base_latched + 8'd2;
                dma_wdata = {4'b0000, result_hold[19:16]};
                job_done  = dma_ready;
            end
            default: ; // IDLE, START_CORE, and WAIT_CORE have no RAM request.
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= DMA_IDLE;
            len_latched    <= 8'd0;
            a_base_latched <= 8'd0;
            b_base_latched <= 8'd0;
            dst_base_latched <= 8'd0;
            pair_index     <= 5'd0;
            a_hold         <= 8'd0;
            b_hold         <= 8'd0;
            result_hold    <= 20'd0;
        end else begin
            case (state)
                DMA_IDLE: begin
                    if (job_start) begin
                        len_latched      <= job_len;
                        a_base_latched   <= job_a_base;
                        b_base_latched   <= job_b_base;
                        dst_base_latched <= job_dst_base;
                        pair_index       <= 5'd0;
                        result_hold      <= 20'd0;
                        state            <= DMA_START_CORE;
                    end
                end

                DMA_START_CORE: begin
                    if (len_latched == 8'd0)
                        state <= DMA_WAIT_CORE;
                    else
                        state <= DMA_READ_A;
                end

                DMA_READ_A: if (dma_ready) begin
                    a_hold <= dma_rdata;
                    state  <= DMA_READ_B;
                end

                DMA_READ_B: if (dma_ready) begin
                    b_hold <= dma_rdata;
                    state  <= DMA_SEND_PAIR;
                end

                DMA_SEND_PAIR: if (s_valid && s_ready) begin
                    if (pair_index == (len_latched[4:0] - 5'd1)) begin
                        state <= DMA_WAIT_CORE;
                    end else begin
                        pair_index <= pair_index + 5'd1;
                        state      <= DMA_READ_A;
                    end
                end

                DMA_WAIT_CORE: if (core_done) begin
                    result_hold <= core_result;
                    state       <= DMA_WRITE_R0;
                end

                DMA_WRITE_R0: if (dma_ready) state <= DMA_WRITE_R1;
                DMA_WRITE_R1: if (dma_ready) state <= DMA_WRITE_R2;
                DMA_WRITE_R2: if (dma_ready) state <= DMA_IDLE;

                default: state <= DMA_IDLE;
            endcase
        end
    end
endmodule
