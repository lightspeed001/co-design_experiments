`timescale 1ns/1ps

// Tiny8 CPU core using the nine-instruction ISA from Foundations 1.
// Unlike the original one-cycle memory sketch, memory instructions retire only
// when dmem_ready is high, so a losing RAM request remains stable during stalls.
module tiny_cpu_wait (
    input  logic        clk,
    input  logic        rst_n,

    // Separate instruction memory: combinational read of a 16-bit word.
    output logic [7:0]  imem_addr,
    input  logic [15:0] imem_rdata,

    // Data-memory/MMIO request channel. Reads are combinational when ready;
    // stores commit at the rising edge on which dmem_ready is asserted.
    output logic        dmem_req,
    output logic        dmem_we,
    output logic [7:0]  dmem_addr,
    output logic [7:0]  dmem_wdata,
    input  logic        dmem_ready,
    input  logic [7:0]  dmem_rdata,

    output logic [7:0]  debug_pc,
    output logic [7:0]  debug_acc,
    output logic        halted
);
    localparam logic [3:0] OP_NOP  = 4'h0;
    localparam logic [3:0] OP_LDI  = 4'h1;
    localparam logic [3:0] OP_LDA  = 4'h2;
    localparam logic [3:0] OP_STA  = 4'h3;
    localparam logic [3:0] OP_ADDI = 4'h4;
    localparam logic [3:0] OP_ADD  = 4'h5;
    localparam logic [3:0] OP_JMP  = 4'h6;
    localparam logic [3:0] OP_JZ   = 4'h7;
    localparam logic [3:0] OP_HALT = 4'hF;

    logic [7:0] pc;
    logic [7:0] acc;
    logic [3:0] opcode;
    logic [7:0] operand;

    assign imem_addr = pc;
    assign opcode    = imem_rdata[15:12];
    assign operand   = imem_rdata[7:0];
    assign dmem_addr = operand;
    assign dmem_wdata = acc;
    assign debug_pc  = pc;
    assign debug_acc = acc;

    // Assert a request only for memory instructions; keep it active until grant.
    assign dmem_req = rst_n && !halted &&
                      ((opcode == OP_LDA) || (opcode == OP_STA) || (opcode == OP_ADD));
    assign dmem_we  = dmem_req && (opcode == OP_STA);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc     <= 8'h00;
            acc    <= 8'h00;
            halted <= 1'b0;
        end else if (!halted) begin
            // A denied RAM request is not retired: PC, accumulator, and request
            // address/data all hold until the arbiter grants the transaction.
            if (dmem_req && !dmem_ready) begin
                pc  <= pc;
                acc <= acc;
            end else begin
                case (opcode)
                    OP_NOP:  pc <= pc + 8'd1;
                    OP_LDI: begin acc <= operand; pc <= pc + 8'd1; end
                    OP_LDA: begin acc <= dmem_rdata; pc <= pc + 8'd1; end
                    OP_STA: pc <= pc + 8'd1; // The wrapper commits the store on this edge.
                    OP_ADDI: begin acc <= acc + operand; pc <= pc + 8'd1; end
                    OP_ADD: begin acc <= acc + dmem_rdata; pc <= pc + 8'd1; end
                    OP_JMP: pc <= operand;
                    OP_JZ:  pc <= (acc == 8'h00) ? operand : (pc + 8'd1);
                    OP_HALT: halted <= 1'b1;
                    default: pc <= pc + 8'd1; // Reserved opcodes behave as NOPs.
                endcase
            end
        end
    end
endmodule
