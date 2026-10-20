`timescale 1ns/1ps

// Two-client, single-port scratchpad with a combinational read and round-robin
// selection under contention. Both clients must keep requests stable until READY.
module ram_arbiter (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        cpu_req,
    input  logic        cpu_we,
    input  logic [7:0]  cpu_addr,
    input  logic [7:0]  cpu_wdata,
    output logic        cpu_ready,
    output logic [7:0]  cpu_rdata,

    input  logic        dma_req,
    input  logic        dma_we,
    input  logic [7:0]  dma_addr,
    input  logic [7:0]  dma_wdata,
    output logic        dma_ready,
    output logic [7:0]  dma_rdata
);
    logic [7:0] ram [0:239];
    logic       prefer_dma;
    logic       grant_cpu;
    logic       grant_dma;
    logic       ram_we;
    logic [7:0] ram_addr;
    logic [7:0] ram_wdata;

    always_comb begin
        grant_cpu = 1'b0;
        grant_dma = 1'b0;
        if (cpu_req && dma_req) begin
            if (prefer_dma) grant_dma = 1'b1;
            else            grant_cpu = 1'b1;
        end else if (cpu_req) begin
            grant_cpu = 1'b1;
        end else if (dma_req) begin
            grant_dma = 1'b1;
        end

        cpu_ready = grant_cpu;
        dma_ready = grant_dma;
        ram_we    = 1'b0;
        ram_addr  = 8'h00;
        ram_wdata = 8'h00;
        if (grant_cpu) begin
            ram_addr  = cpu_addr;
            ram_wdata = cpu_wdata;
            ram_we    = cpu_we;
        end else if (grant_dma) begin
            ram_addr  = dma_addr;
            ram_wdata = dma_wdata;
            ram_we    = dma_we;
        end

        // Only the owner of this cycle observes the selected RAM byte.
        cpu_rdata = grant_cpu ? ram[ram_addr] : 8'h00;
        dma_rdata = grant_dma ? ram[ram_addr] : 8'h00;
    end

    // A selected store commits on its grant edge; ungranted writes have no effect.
    always_ff @(posedge clk) begin
        if (ram_we)
            ram[ram_addr] <= ram_wdata;
    end

    // Alternating preference prevents either persistent requester from starving.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            prefer_dma <= 1'b0;
        else if (cpu_req && dma_req)
            prefer_dma <= !prefer_dma;
    end
endmodule
