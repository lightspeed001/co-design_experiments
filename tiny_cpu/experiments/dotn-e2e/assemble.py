#!/usr/bin/env python3
"""Assemble the small, fixed-width Tiny8 test program used by the RTL run.

This deliberately tiny two-pass assembler supports labels, the nine teaching
ISA mnemonics, decimal/hex byte operands, and '#' on immediate operands. It is
not the Toy8 compiler from Foundations 2.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

OPCODES = {
    "NOP": 0x0,
    "LDI": 0x1,
    "LDA": 0x2,
    "STA": 0x3,
    "ADDI": 0x4,
    "ADD": 0x5,
    "JMP": 0x6,
    "JZ": 0x7,
    "HALT": 0xF,
}
NO_OPERAND = {"NOP", "HALT"}
IMMEDIATE = {"LDI", "ADDI"}


def parse_byte(token: str, labels: dict[str, int], line_no: int) -> int:
    """Resolve one byte literal or instruction-word label."""
    token = token.strip()
    if token in labels:
        value = labels[token]
    else:
        try:
            value = int(token, 0)
        except ValueError as exc:
            raise ValueError(f"line {line_no}: unknown label or byte literal {token!r}") from exc
    if not 0 <= value <= 0xFF:
        raise ValueError(f"line {line_no}: operand {value} is outside the 8-bit range")
    return value


def source_lines(path: Path) -> list[tuple[int, str]]:
    """Return nonempty source statements with their original line numbers."""
    statements = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        statement = raw.split(";", 1)[0].strip()
        if statement:
            statements.append((line_no, statement))
    return statements


def assemble(source: Path) -> list[int]:
    """Resolve labels first, then encode each instruction as opcode|operand."""
    statements = source_lines(source)
    labels: dict[str, int] = {}
    instructions: list[tuple[int, str, str]] = []
    pc = 0

    # Pass one records labels and instruction-word positions.
    for line_no, statement in statements:
        if ":" in statement:
            label, remainder = statement.split(":", 1)
            label = label.strip()
            if not label or label in labels:
                raise ValueError(f"line {line_no}: empty or duplicate label {label!r}")
            labels[label] = pc
            statement = remainder.strip()
            if not statement:
                continue

        fields = statement.split(None, 1)
        mnemonic = fields[0].upper()
        operand = fields[1].strip() if len(fields) == 2 else ""
        if mnemonic not in OPCODES:
            raise ValueError(f"line {line_no}: unsupported mnemonic {mnemonic!r}")
        if (mnemonic in NO_OPERAND) != (not operand):
            expected = "no operand" if mnemonic in NO_OPERAND else "one operand"
            raise ValueError(f"line {line_no}: {mnemonic} expects {expected}")
        instructions.append((line_no, mnemonic, operand))
        pc += 1

    if pc > 256:
        raise ValueError("program exceeds the 256-word instruction memory")

    # Pass two resolves labels and writes the 4-bit opcode plus 8-bit operand.
    words = []
    for line_no, mnemonic, operand in instructions:
        if mnemonic in NO_OPERAND:
            value = 0
        else:
            if mnemonic in IMMEDIATE:
                if not operand.startswith("#"):
                    raise ValueError(f"line {line_no}: {mnemonic} requires an immediate marked with '#'")
                operand = operand[1:].strip()
            elif operand.startswith("#"):
                raise ValueError(f"line {line_no}: {mnemonic} uses an address/label, not an immediate")
            value = parse_byte(operand, labels, line_no)
        words.append((OPCODES[mnemonic] << 12) | value)
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Tiny8 assembly source")
    parser.add_argument("output", type=Path, help="output $readmemh-compatible hex file")
    args = parser.parse_args()
    try:
        words = assemble(args.source)
    except (OSError, ValueError) as exc:
        print(f"assemble: {exc}", file=sys.stderr)
        return 1

    instruction_count = len(words)
    rom_image = words + [0xF000] * (256 - instruction_count)  # Unused words halt safely.
    args.output.write_text("".join(f"{word:04X}\n" for word in rom_image), encoding="ascii")
    print(f"assembled {instruction_count} instructions into a 256-word ROM image: {args.source} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
