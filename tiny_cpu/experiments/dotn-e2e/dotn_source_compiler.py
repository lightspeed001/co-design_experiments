#!/usr/bin/env python3
"""Compile a deliberately restricted dot_u8_into source file to Tiny8 assembly.

This executable prototype demonstrates one target-specific lowering path. It is
not the Rust/Melior Toy8 compiler: the accepted language is only u8 array
literals, one dot_u8_into call, and one expected-result check.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

DECL_RE = re.compile(
    r"^let\s+(\w+)\s*:\s*\[u8\s*;\s*(\d+)\s*\]\s*=\s*\[([^\]]*)\]\s*;?$"
)
CALL_RE = re.compile(
    r"^dot_u8_into\(\s*(\w+)\s*,\s*(\w+)\s*,\s*len\s*=\s*(\d+)\s*,\s*dst\s*=\s*(\w+)\s*\)\s*;?$"
)
EXPECT_RE = re.compile(r"^expect\s+(\w+)\s*==\s*\[([^\]]*)\]\s*;?$")

# Fixed target layout used by this one-operation teaching backend.
A_BASE = 0x20
B_BASE = 0x30
DST_BASE = 0x40
SCRATCH_ADDR = 0x60
PASS_ADDR = 0x50
MAX_TERMS = 16
RAM_END = 0xF0  # Exclusive; ordinary scratchpad is 0x00 through 0xEF.


@dataclass(frozen=True)
class U8Array:
    """A statically sized u8 array declaration."""
    name: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class Program:
    """The small typed source program after parsing and semantic checks."""
    arrays: dict[str, U8Array]
    a_name: str
    b_name: str
    result_name: str
    length: int
    expected: tuple[int, int, int]


def parse_u8_list(text: str, line_no: int) -> tuple[int, ...]:
    """Parse comma-separated decimal/hex byte literals and check their width."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    try:
        values = tuple(int(part, 0) for part in parts)
    except ValueError as exc:
        raise ValueError(f"line {line_no}: expected decimal or 0x-prefixed byte literals") from exc
    for value in values:
        if not 0 <= value <= 0xFF:
            raise ValueError(f"line {line_no}: {value} is outside the u8 range")
    return values


def parse_source(path: Path) -> Program:
    """Parse the intentionally small source grammar and reject unsupported syntax."""
    arrays: dict[str, U8Array] = {}
    call: tuple[str, str, int, str, int] | None = None
    expectation: tuple[str, tuple[int, ...], int] | None = None

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        # This teaching parser accepts // line comments only.
        statement = raw.split("//", 1)[0].strip()
        if not statement:
            continue

        if match := DECL_RE.fullmatch(statement):
            name, length_text, values_text = match.groups()
            if name in arrays:
                raise ValueError(f"line {line_no}: duplicate declaration {name!r}")
            declared_len = int(length_text)
            values = parse_u8_list(values_text, line_no)
            if len(values) != declared_len:
                raise ValueError(f"line {line_no}: declared length {declared_len}, got {len(values)} initializer bytes")
            arrays[name] = U8Array(name, values)
            continue

        if match := CALL_RE.fullmatch(statement):
            if call is not None:
                raise ValueError(f"line {line_no}: only one dot_u8_into call is supported")
            a_name, b_name, len_text, result_name = match.groups()
            call = (a_name, b_name, int(len_text), result_name, line_no)
            continue

        if match := EXPECT_RE.fullmatch(statement):
            if expectation is not None:
                raise ValueError(f"line {line_no}: only one expect statement is supported")
            result_name, values_text = match.groups()
            expectation = (result_name, parse_u8_list(values_text, line_no), line_no)
            continue

        raise ValueError(f"line {line_no}: unsupported source statement: {statement}")

    if call is None:
        raise ValueError("source needs one dot_u8_into call")
    if expectation is None:
        raise ValueError("source needs one expected three-byte result")

    a_name, b_name, length, result_name, call_line = call
    expected_name, expected_values, expect_line = expectation
    for name in (a_name, b_name, result_name):
        if name not in arrays:
            raise ValueError(f"line {call_line}: undeclared array {name!r}")
    if expected_name != result_name:
        raise ValueError(f"line {expect_line}: expectation must check the destination array {result_name!r}")
    if len(expected_values) != 3:
        raise ValueError(f"line {expect_line}: expected result must contain exactly three bytes")
    if not 0 <= length <= MAX_TERMS:
        raise ValueError(f"line {call_line}: LEN must be in 0..={MAX_TERMS}")
    if length > len(arrays[a_name].values) or length > len(arrays[b_name].values):
        raise ValueError(f"line {call_line}: LEN exceeds an input array's declared length")
    if len(arrays[result_name].values) != 3:
        raise ValueError(f"line {call_line}: destination must be exactly three bytes")
    if result_name in (a_name, b_name):
        raise ValueError(f"line {call_line}: this fixed-layout prototype requires a distinct destination array")

    # Check that fixed input/output slots cannot overlap before emitting instructions.
    a_end = A_BASE + len(arrays[a_name].values)
    b_end = B_BASE + len(arrays[b_name].values)
    if a_end > B_BASE:
        raise ValueError(f"array {a_name!r} overlaps the fixed B array slot")
    if b_end > DST_BASE:
        raise ValueError(f"array {b_name!r} overlaps the fixed result slot")
    if DST_BASE + 3 > RAM_END or SCRATCH_ADDR >= RAM_END or PASS_ADDR >= RAM_END:
        raise ValueError("fixed target layout exceeds the scratchpad")

    # The source's expected bytes are checked against a wide compile-time evaluator.
    wide_sum = sum(arrays[a_name].values[i] * arrays[b_name].values[i] for i in range(length))
    if wide_sum >= (1 << 20):
        raise ValueError("dot_u8_into result does not fit the 20-bit hardware contract")
    result_bytes = (wide_sum & 0xFF, (wide_sum >> 8) & 0xFF, (wide_sum >> 16) & 0x0F)
    if tuple(expected_values) != result_bytes:
        raise ValueError(
            f"line {expect_line}: expected bytes {tuple(expected_values)!r}, "
            f"but compile-time dot result is {result_bytes!r}"
        )

    return Program(arrays, a_name, b_name, result_name, length, result_bytes)


def emit_assembly(program: Program) -> str:
    """Lower this source slice to CPU setup, MMIO, polling, and result-check code."""
    lines = [
        "; Generated by dotn_source_compiler.py from dotn_call.toy8.",
        "; Fixed Tiny8 memory map: A=0x20, B=0x30, result=0x40, scratch=0x60.",
        "; The output buffer is initialized before the asynchronous device call.",
    ]

    # Materialize source arrays and output initialization in ordinary data RAM.
    for array_name, base in ((program.a_name, A_BASE), (program.b_name, B_BASE), (program.result_name, DST_BASE)):
        for offset, value in enumerate(program.arrays[array_name].values):
            lines.append(f"LDI  #0x{value:02X}")
            lines.append(f"STA  0x{base + offset:02X}")
    lines.extend([
        "LDI  #0x00",
        f"STA  0x{SCRATCH_ADDR:02X}       ; Scratch read in poll loop creates CPU/DMA RAM contention.",
        "",
        "; Emit the target-specific descriptor and START write for dot_u8_into.",
        f"LDI  #0x{program.length:02X}",
        "STA  0xF2       ; LEN",
        f"LDI  #0x{A_BASE:02X}",
        "STA  0xF7       ; A_BASE",
        f"LDI  #0x{B_BASE:02X}",
        "STA  0xF8       ; B_BASE",
        f"LDI  #0x{DST_BASE:02X}",
        "STA  0xF9       ; DST_BASE",
        "LDI  #0x01",
        "STA  0xF0       ; START",
        "",
        "poll:",
        f"LDA  0x{SCRATCH_ADDR:02X}       ; Ordinary RAM read can contend with DMA.",
        "LDA  0xF1       ; DONE",
        "JZ   poll",
        "",
        "; Check ERROR and each little-endian result byte on the Tiny8 CPU.",
        "LDA  0xF3",
        "JZ   check_byte_0",
        "JMP  fail",
    ])

    for index, expected_byte in enumerate(program.expected):
        lines.append(f"check_byte_{index}:")
        lines.append(f"LDA  0x{DST_BASE + index:02X}")
        lines.append(f"ADDI #0x{(-expected_byte) & 0xFF:02X} ; Zero only when byte {index} matches.")
        next_label = f"check_byte_{index + 1}" if index < 2 else "pass"
        lines.append(f"JZ   {next_label}")
        lines.append("JMP  fail")

    lines.extend([
        "",
        "pass:",
        f"LDI  #0xA5",
        f"STA  0x{PASS_ADDR:02X}",
        "HALT",
        "",
        "fail:",
        f"LDI  #0xEE",
        f"STA  0x{PASS_ADDR:02X}",
        "HALT",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="restricted .toy8 source file")
    parser.add_argument("output", type=Path, help="generated Tiny8 assembly file")
    args = parser.parse_args()
    try:
        program = parse_source(args.source)
        assembly = emit_assembly(program)
        args.output.write_text(assembly, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"dotn-source-compiler: {exc}", file=sys.stderr)
        return 1

    print(
        f"compiled dot_u8_into({program.a_name}, {program.b_name}, "
        f"LEN={program.length}, {program.result_name}) -> {args.output}"
    )
    print(f"fixed target placement: A=0x{A_BASE:02X}, B=0x{B_BASE:02X}, result=0x{DST_BASE:02X}")
    print(f"compile-time checked result: {program.expected[0]:02X} {program.expected[1]:02X} {program.expected[2]:02X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
