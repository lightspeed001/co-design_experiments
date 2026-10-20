`timescale 1ns/1ps

// CPU-visible MMIO wrapper around the DMA, single-port RAM arbiter, and dotN MAC.
// The system controller snapshots and validates the descriptor before launch;
// DONE rises only on the edge that commits the third result byte to scratchpad.
module dotn_system (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        cpu_req,
    input  logic        cpu_we,
    input  logic [7:0]  cpu_addr,
    input  logic [7:0]  cpu_wdata,
    output logic        cpu_ready,
    output logic [7:0]  cpu_rdata,

    output logic        done,
    output logic        error
);
    localparam logic [7:0] REG_START = 8'hF0;
    localparam logic [7:0] REG_DONE  = 8'hF1;
    localparam logic [7:0] REG_LEN   = 8'hF2;
    localparam logic [7:0] REG_ERROR = 8'hF3;
    localparam logic [7:0] REG_RES0  = 8'hF4;
    localparam logic [7:0] REG_RES1  = 8'hF5;
    localparam logic [7:0] REG_RES2  = 8'hF6;
    localparam logic [7:0] REG_A     = 8'hF7;
    localparam logic [7:0] REG_B     = 8'hF8;
    localparam logic [7:0] REG_DST   = 8'hF9;

    logic [7:0] len_reg;
    logic [7:0] a_base_reg;
    logic [7:0] b_base_reg;
    logic [7:0] dst_base_reg;
    logic [7:0] result0;
    logic [7:0] result1;
    logic [7:0] result2;
    logic [7:0] cfg_rdata;

    logic [8:0] a_end;
    logic [8:0] b_end;
    logic [8:0] dst_end;
    logic       descriptor_valid;
    logic       launch_pending;
    logic       system_busy;

    logic        cpu_ram_req;
    logic        cpu_ram_ready;
    logic [7:0]  cpu_ram_rdata;
    logic        dma_req;
    logic        dma_we;
    logic [7:0]  dma_addr;
    logic [7:0]  dma_wdata;
    logic        dma_ready;
    logic [7:0]  dma_rdata;
    logic        dma_job_done;
    logic        dma_busy;
    logic [19:0] dma_result;

    logic        core_job_start;
    logic [4:0]  core_job_len;
    logic        s_valid;
    logic        s_ready;
    logic [7:0]  s_a;
    logic [7:0]  s_b;
    logic        core_done;
    logic [19:0] core_result;

    // Widen before adding so 8-bit address wrap cannot validate a bad range.
    always_comb begin
        a_end   = {1'b0, a_base_reg} + {1'b0, len_reg};
        b_end   = {1'b0, b_base_reg} + {1'b0, len_reg};
        dst_end = {1'b0, dst_base_reg} + 9'd3;
        descriptor_valid = (len_reg <= 8'd16) &&
                           ((len_reg == 8'd0) ||
                            ((a_end <= 9'd240) && (b_end <= 9'd240))) &&
                           (dst_end <= 9'd240);
    end

    assign cpu_ram_req = cpu_req && (cpu_addr <= 8'hEF);
    assign cpu_ready   = (cpu_addr <= 8'hEF) ? cpu_ram_ready : 1'b1;
    assign cpu_rdata   = (cpu_addr <= 8'hEF) ? cpu_ram_rdata :
                         ((cpu_addr >= 8'hF0 && cpu_addr <= 8'hFD) ? cfg_rdata : 8'h00);
    assign system_busy = launch_pending || dma_busy;
    assign done         = done_reg;
    assign error        = error_reg;

    logic done_reg;
    logic error_reg;

    // Status and configuration reads are combinational; reserved/unmapped bytes read zero.
    always_comb begin
        cfg_rdata = 8'h00;
        case (cpu_addr)
            REG_DONE:  cfg_rdata = {7'b0, done_reg};
            REG_LEN:   cfg_rdata = len_reg;
            REG_ERROR: cfg_rdata = {7'b0, error_reg};
            REG_RES0:  cfg_rdata = result0;
            REG_RES1:  cfg_rdata = result1;
            REG_RES2:  cfg_rdata = result2;
            REG_A:     cfg_rdata = a_base_reg;
            REG_B:     cfg_rdata = b_base_reg;
            REG_DST:   cfg_rdata = dst_base_reg;
            default:   cfg_rdata = 8'h00;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            len_reg       <= 8'd0;
            a_base_reg    <= 8'd0;
            b_base_reg    <= 8'd0;
            dst_base_reg  <= 8'd0;
            result0       <= 8'd0;
            result1       <= 8'd0;
            result2       <= 8'd0;
            done_reg      <= 1'b0;
            error_reg     <= 1'b0;
            launch_pending <= 1'b0;
        end else begin
            // This delayed pulse lets the descriptor snapshots settle before DMA accepts them.
            launch_pending <= 1'b0;

            if (dma_job_done) begin
                result0   <= dma_result[7:0];
                result1   <= dma_result[15:8];
                result2   <= {4'b0000, dma_result[19:16]};
                done_reg  <= 1'b1;
                error_reg <= 1'b0;
            end

            if (cpu_req && cpu_we && !system_busy) begin
                case (cpu_addr)
                    REG_LEN: len_reg <= cpu_wdata;
                    REG_A:   a_base_reg <= cpu_wdata;
                    REG_B:   b_base_reg <= cpu_wdata;
                    REG_DST: dst_base_reg <= cpu_wdata;
                    REG_START: if (cpu_wdata[0]) begin
                        // Snapshot the just-programmed descriptor on the accepted START edge.
                        result0  <= 8'd0;
                        result1  <= 8'd0;
                        result2  <= 8'd0;
                        done_reg <= 1'b0;
                        error_reg <= 1'b0;
                        if (descriptor_valid) begin
                            launch_pending <= 1'b1;
                        end else begin
                            done_reg  <= 1'b1;
                            error_reg <= 1'b1;
                        end
                    end
                    default: ; // Status and result registers are read-only.
                endcase
            end
        end
    end

    ram_arbiter u_arbiter (
        .clk(clk), .rst_n(rst_n),
        .cpu_req(cpu_ram_req), .cpu_we(cpu_we), .cpu_addr(cpu_addr),
        .cpu_wdata(cpu_wdata), .cpu_ready(cpu_ram_ready), .cpu_rdata(cpu_ram_rdata),
        .dma_req(dma_req), .dma_we(dma_we), .dma_addr(dma_addr),
        .dma_wdata(dma_wdata), .dma_ready(dma_ready), .dma_rdata(dma_rdata)
    );

    dotn_dma u_dma (
        .clk(clk), .rst_n(rst_n), .job_start(launch_pending),
        .job_len(len_reg), .job_a_base(a_base_reg), .job_b_base(b_base_reg),
        .job_dst_base(dst_base_reg),
        .dma_req(dma_req), .dma_we(dma_we), .dma_addr(dma_addr),
        .dma_wdata(dma_wdata), .dma_ready(dma_ready), .dma_rdata(dma_rdata),
        .core_job_start(core_job_start), .core_job_len(core_job_len),
        .s_valid(s_valid), .s_ready(s_ready), .s_a(s_a), .s_b(s_b),
        .core_done(core_done), .core_result(core_result),
        .job_done(dma_job_done), .busy(dma_busy), .result(dma_result)
    );

    dotn_mac_core u_core (
        .clk(clk), .rst_n(rst_n), .job_start(core_job_start), .job_len(core_job_len),
        .s_valid(s_valid), .s_ready(s_ready), .s_a(s_a), .s_b(s_b),
        .core_done(core_done), .core_result(core_result)
    );
endmodule
