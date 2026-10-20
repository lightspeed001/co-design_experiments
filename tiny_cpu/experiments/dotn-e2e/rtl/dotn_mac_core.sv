`timescale 1ns/1ps

// Unsigned ready/valid dotN core. One accepted pair contributes one product.
module dotn_mac_core (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        job_start,
    input  logic [4:0]  job_len,       // Validated range is 0..16.
    input  logic        s_valid,
    output logic        s_ready,
    input  logic [7:0]  s_a,
    input  logic [7:0]  s_b,
    output logic        core_done,     // One-cycle pulse; result remains stable afterward.
    output logic [19:0] core_result
);
    typedef enum logic { ST_IDLE, ST_RUN } state_t;
    state_t state;

    logic [4:0]  len_latched;
    logic [4:0]  count;
    logic [19:0] accumulator;
    logic [15:0] product;
    logic [19:0] accumulator_next;

    assign s_ready = (state == ST_RUN);

    // Keep the product full-width before adding it into the 20-bit sum.
    always_comb begin
        product         = {8'h00, s_a} * {8'h00, s_b};
        accumulator_next = accumulator + {4'b0000, product};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            len_latched <= 5'd0;
            count       <= 5'd0;
            accumulator <= 20'd0;
            core_result <= 20'd0;
            core_done   <= 1'b0;
        end else begin
            core_done <= 1'b0;
            case (state)
                ST_IDLE: begin
                    if (job_start) begin
                        len_latched <= job_len;
                        count       <= 5'd0;
                        accumulator <= 20'd0;
                        core_result <= 20'd0;
                        if (job_len == 5'd0 || job_len > 5'd16) begin
                            // Empty jobs return zero; invalid lengths fail closed.
                            core_result <= 20'd0;
                            core_done   <= 1'b1;
                        end else begin
                            state <= ST_RUN;
                        end
                    end
                end

                ST_RUN: begin
                    if (s_valid && s_ready) begin
                        accumulator <= accumulator_next;
                        if (count == (len_latched - 5'd1)) begin
                            // Publish the next sum so the final accepted term is included.
                            core_result <= accumulator_next;
                            core_done   <= 1'b1;
                            state       <= ST_IDLE;
                        end else begin
                            count <= count + 5'd1;
                        end
                    end
                end

                default: begin
                    state       <= ST_IDLE;
                    count       <= 5'd0;
                    accumulator <= 20'd0;
                    core_result <= 20'd0;
                    core_done   <= 1'b0;
                end
            endcase
        end
    end
endmodule
