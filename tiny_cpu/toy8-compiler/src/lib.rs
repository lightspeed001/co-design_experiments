pub mod diagnostic;
pub mod frontend;
mod mlir_backend;
pub mod target;

use crate::{diagnostic::Diagnostic, frontend::Program, target::Instruction};

/// Artifacts produced from one valid Toy8 v1 source program.
#[derive(Debug, Clone)]
pub struct Compilation {
    pub program: Program,
    /// Printed, verifier-accepted MLIR module.
    pub mlir: String,
    /// Tiny8 instructions selected by traversing the verified MLIR module.
    pub instructions: Vec<Instruction>,
    /// Complete 256-word ROM, padded with HALT instructions.
    pub words: Vec<u16>,
    /// `$readmemh`-compatible, one-16-bit-word-per-line image.
    pub hex: String,
    /// Human-readable address/mnemonic/encoding listing.
    pub assembly: String,
    /// Expected `main` result from an independent wrapping-u8 AST evaluator.
    pub expected_return: u8,
}

/// Compile a source string through parsing, semantic constraints, Melior IR
/// construction and verification, MLIR operation selection, and Tiny8 encoding.
pub fn compile_source(source: &str) -> Result<Compilation, Diagnostic> {
    let program = frontend::parse_source(source)?;
    let expected_return = program.evaluate();
    let (mlir, instructions) = mlir_backend::compile_mlir_to_tiny8(&program)?;
    let words = target::rom_image(&instructions)?;
    let hex = target::hex_image(&words);
    let assembly = target::assembly_listing(&instructions);

    Ok(Compilation {
        program,
        mlir,
        instructions,
        words,
        hex,
        assembly,
        expected_return,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_the_foundations_two_example_through_melior_and_tiny8() {
        let compilation = compile_source("fn main() -> u8 { return 2 + 3; }").unwrap();
        assert!(compilation.mlir.contains("func.func @main() -> i8"));
        assert!(compilation.mlir.contains("arith.constant 2 : i8"));
        assert!(compilation.mlir.contains("arith.constant 3 : i8"));
        assert!(compilation.mlir.contains("arith.addi"));
        assert_eq!(compilation.expected_return, 5);
        assert_eq!(
            compilation.instructions,
            vec![
                Instruction::Ldi(2),
                Instruction::Addi(3),
                Instruction::Sta(0x10),
                Instruction::Halt,
            ]
        );
        assert_eq!(&compilation.words[..4], &[0x1002, 0x4003, 0x3010, 0xF000]);
    }

    #[test]
    fn compiles_single_literal_without_emitting_addi() {
        let compilation = compile_source("fn main() -> u8 { return 0xA5; }").unwrap();
        assert_eq!(compilation.expected_return, 0xA5);
        assert_eq!(
            compilation.instructions,
            vec![
                Instruction::Ldi(0xA5),
                Instruction::Sta(0x10),
                Instruction::Halt
            ]
        );
    }

    #[test]
    fn compiles_left_associative_sum_and_wraps_like_u8() {
        let compilation = compile_source("fn main() -> u8 { return 255 + 1 + 2; }").unwrap();
        assert_eq!(compilation.expected_return, 2);
        assert_eq!(
            compilation.instructions,
            vec![
                Instruction::Ldi(255),
                Instruction::Addi(1),
                Instruction::Addi(2),
                Instruction::Sta(0x10),
                Instruction::Halt,
            ]
        );
    }
}
