"""Unit tests for the restricted source-to-Tiny8 lowering prototype."""
from pathlib import Path
import tempfile
import unittest

from dotn_source_compiler import emit_assembly, parse_source


class DotnSourceCompilerTests(unittest.TestCase):
    def compile_text(self, text: str):
        """Write a temporary source file and run the real parser/type/layout checks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "case.toy8"
            source.write_text(text, encoding="utf-8")
            return parse_source(source)

    def test_reference_program_lowers_to_tiny8(self):
        """The checked-in source lowers its descriptor and CPU-side DONE poll."""
        program = parse_source(Path("dotn_call.toy8"))
        assembly = emit_assembly(program)
        self.assertEqual(program.length, 4)
        self.assertEqual(program.expected, (0x46, 0x00, 0x00))
        self.assertIn("STA  0xF0       ; START", assembly)
        self.assertIn("LDA  0xF1       ; DONE", assembly)
        self.assertIn("STA  0x50", assembly)

    def test_rejects_unsupported_length(self):
        """The hardware limit is enforced before any assembly is emitted."""
        source = """\
let A: [u8; 1] = [1];
let B: [u8; 1] = [1];
let result: [u8; 3] = [0, 0, 0];
dot_u8_into(A, B, len=17, dst=result);
expect result == [1, 0, 0];
"""
        with self.assertRaisesRegex(ValueError, "LEN must be in 0..=16"):
            self.compile_text(source)

    def test_rejects_wrong_compile_time_expected_bytes(self):
        """A stale expected result must not silently become the CPU assertion."""
        source = """\
let A: [u8; 1] = [1];
let B: [u8; 1] = [2];
let result: [u8; 3] = [0, 0, 0];
dot_u8_into(A, B, len=1, dst=result);
expect result == [1, 0, 0];
"""
        with self.assertRaisesRegex(ValueError, "compile-time dot result"):
            self.compile_text(source)

    def test_rejects_fixed_layout_overlap(self):
        """Array extents may not collide with the next fixed target slot."""
        a_values = ", ".join(["0"] * 17)
        source = f"""\
let A: [u8; 17] = [{a_values}];
let B: [u8; 1] = [0];
let result: [u8; 3] = [0, 0, 0];
dot_u8_into(A, B, len=1, dst=result);
expect result == [0, 0, 0];
"""
        with self.assertRaisesRegex(ValueError, "overlaps the fixed B array slot"):
            self.compile_text(source)


if __name__ == "__main__":
    unittest.main()
