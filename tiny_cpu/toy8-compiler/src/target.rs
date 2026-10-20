use crate::diagnostic::Diagnostic;

pub const ROM_WORDS: usize = 256;

/// The nine instructions defined by Foundations 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    Nop,
    Ldi(u8),
    Lda(u8),
    Sta(u8),
    Addi(u8),
    Add(u8),
    Jmp(u8),
    Jz(u8),
    Halt,
}

impl Instruction {
    /// Encode `[opcode:4][reserved:4][operand:8]`, with reserved bits zero.
    pub const fn encode(self) -> u16 {
        let (opcode, operand) = match self {
            Self::Nop => (0x0, 0x00),
            Self::Ldi(value) => (0x1, value),
            Self::Lda(address) => (0x2, address),
            Self::Sta(address) => (0x3, address),
            Self::Addi(value) => (0x4, value),
            Self::Add(address) => (0x5, address),
            Self::Jmp(target) => (0x6, target),
            Self::Jz(target) => (0x7, target),
            Self::Halt => (0xF, 0x00),
        };
        ((opcode as u16) << 12) | operand as u16
    }

    pub fn assembly(self) -> String {
        match self {
            Self::Nop => "NOP".to_owned(),
            Self::Ldi(value) => format!("LDI  #0x{value:02X}"),
            Self::Lda(address) => format!("LDA  0x{address:02X}"),
            Self::Sta(address) => format!("STA  0x{address:02X}"),
            Self::Addi(value) => format!("ADDI #0x{value:02X}"),
            Self::Add(address) => format!("ADD  0x{address:02X}"),
            Self::Jmp(target) => format!("JMP  0x{target:02X}"),
            Self::Jz(target) => format!("JZ   0x{target:02X}"),
            Self::Halt => "HALT".to_owned(),
        }
    }
}

/// Create a complete ROM image padded with HALT words so fall-through is safe.
pub fn rom_image(instructions: &[Instruction]) -> Result<Vec<u16>, Diagnostic> {
    if instructions.len() > ROM_WORDS {
        return Err(Diagnostic::global(format!(
            "generated program has {} instructions; Tiny8 ROM holds at most {ROM_WORDS}",
            instructions.len()
        )));
    }

    let mut words = vec![Instruction::Halt.encode(); ROM_WORDS];
    for (index, instruction) in instructions.iter().enumerate() {
        words[index] = instruction.encode();
    }
    Ok(words)
}

pub fn hex_image(words: &[u16]) -> String {
    words.iter().map(|word| format!("{word:04X}\n")).collect()
}

pub fn assembly_listing(instructions: &[Instruction]) -> String {
    let mut listing = String::from("; Generated Tiny8 instruction listing\n");
    for (address, instruction) in instructions.iter().copied().enumerate() {
        listing.push_str(&format!(
            "{address:02X}: {:<13} ; {:04X}\n",
            instruction.assembly(),
            instruction.encode()
        ));
    }
    listing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_nine_defined_instructions_match_the_isa_encodings() {
        let instructions = [
            Instruction::Nop,
            Instruction::Ldi(0x02),
            Instruction::Lda(0x10),
            Instruction::Sta(0x10),
            Instruction::Addi(0x03),
            Instruction::Add(0x10),
            Instruction::Jmp(0x20),
            Instruction::Jz(0x21),
            Instruction::Halt,
        ];
        let expected = [
            0x0000, 0x1002, 0x2010, 0x3010, 0x4003, 0x5010, 0x6020, 0x7021, 0xF000,
        ];
        assert_eq!(
            instructions.map(Instruction::encode),
            expected,
            "the target encoder must preserve the full nine-instruction Foundations 1 ISA"
        );
    }

    #[test]
    fn return_two_plus_three_encodes_to_the_hand_traced_sequence() {
        let instructions = [
            Instruction::Ldi(2),
            Instruction::Addi(3),
            Instruction::Sta(0x10),
            Instruction::Halt,
        ];
        assert_eq!(
            instructions.map(Instruction::encode),
            [0x1002, 0x4003, 0x3010, 0xF000]
        );
    }

    #[test]
    fn rom_image_is_256_words_and_fills_unused_words_with_halt() {
        let words = rom_image(&[
            Instruction::Ldi(5),
            Instruction::Sta(0x10),
            Instruction::Halt,
        ])
        .unwrap();
        assert_eq!(words.len(), ROM_WORDS);
        assert_eq!(&words[..3], &[0x1005, 0x3010, 0xF000]);
        assert!(words[3..].iter().all(|word| *word == 0xF000));
        assert_eq!(hex_image(&words).lines().count(), ROM_WORDS);
    }

    #[test]
    fn assembly_listing_includes_addresses_and_words() {
        let listing = assembly_listing(&[Instruction::Ldi(2), Instruction::Halt]);
        assert!(listing.contains("00: LDI  #0x02"));
        assert!(listing.contains("1002"));
        assert!(listing.contains("01: HALT"));
        assert!(listing.contains("F000"));
    }
}
