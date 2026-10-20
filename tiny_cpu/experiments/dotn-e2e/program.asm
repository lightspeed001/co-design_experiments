; End-to-end directed program for the Tiny8 ISA.
; Create A=[1,2,3,4] at 0x20 and B=[5,6,7,8] at 0x30.
; Submit dotN(LEN=4) with output at 0x40, poll DONE, verify 0x00046,
; then write 0xA5 (pass) or 0xEE (fail) to 0x50 before HALT.

LDI  #0x01
STA  0x20
LDI  #0x02
STA  0x21
LDI  #0x03
STA  0x22
LDI  #0x04
STA  0x23

LDI  #0x05
STA  0x30
LDI  #0x06
STA  0x31
LDI  #0x07
STA  0x32
LDI  #0x08
STA  0x33

; This scratch read in the polling loop intentionally competes with DMA for RAM.
LDI  #0x00
STA  0x60

; Program the descriptor: four pairs, two source addresses, and one destination.
LDI  #0x04
STA  0xF2
LDI  #0x20
STA  0xF7
LDI  #0x30
STA  0xF8
LDI  #0x40
STA  0xF9
LDI  #0x01
STA  0xF0

poll:
LDA  0x60
LDA  0xF1
JZ   poll

; Reject ERROR and verify the three little-endian result bytes in RAM.
LDA  0xF3
JZ   check_low
JMP  fail

check_low:
LDA  0x40
ADDI #0xBA
JZ   check_high1
JMP  fail

check_high1:
LDA  0x41
JZ   check_high2
JMP  fail

check_high2:
LDA  0x42
JZ   pass
JMP  fail

pass:
LDI  #0xA5
STA  0x50
HALT

fail:
LDI  #0xEE
STA  0x50
HALT
