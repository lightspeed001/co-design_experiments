use crate::diagnostic::{Diagnostic, Span};

const MAX_TERMS: usize = 254; // LDI + (N-1) ADDI + STA + HALT fits in 256 ROM words.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteLiteral {
    pub value: u8,
    pub span: Span,
}

/// A type-checked v1 Toy8 program: one `main() -> u8` return sum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub literals: Vec<ByteLiteral>,
}

impl Program {
    /// Evaluate with Toy8's wrapping 8-bit addition semantics.
    pub fn evaluate(&self) -> u8 {
        self.literals
            .iter()
            .fold(0u8, |sum, literal| sum.wrapping_add(literal.value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenKind {
    Fn,
    Return,
    U8,
    Identifier(String),
    Number(String),
    Arrow,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Plus,
    Semicolon,
    Eof,
}

impl TokenKind {
    fn description(&self) -> String {
        match self {
            Self::Fn => "`fn`".into(),
            Self::Return => "`return`".into(),
            Self::U8 => "`u8`".into(),
            Self::Identifier(name) => format!("identifier `{name}`"),
            Self::Number(number) => format!("integer literal `{number}`"),
            Self::Arrow => "`->`".into(),
            Self::LeftParen => "`(`".into(),
            Self::RightParen => "`)`".into(),
            Self::LeftBrace => "`{`".into(),
            Self::RightBrace => "`}`".into(),
            Self::Plus => "`+`".into(),
            Self::Semicolon => "`;`".into(),
            Self::Eof => "end of file".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    kind: TokenKind,
    span: Span,
}

struct Lexer<'a> {
    source: &'a str,
    offset: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            offset: 0,
            line: 1,
            column: 1,
        }
    }

    fn current_span(&self) -> Span {
        Span::new(self.line, self.column)
    }

    fn rest(&self) -> &'a str {
        &self.source[self.offset..]
    }

    fn peek_char(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn peek_second_char(&self) -> Option<char> {
        self.rest().chars().nth(1)
    }

    fn bump_char(&mut self) -> Option<char> {
        let character = self.peek_char()?;
        self.offset += character.len_utf8();
        if character == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(character)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek_char() {
                Some(character) if character.is_ascii_whitespace() => {
                    self.bump_char();
                }
                Some('/') if self.rest().starts_with("//") => {
                    self.bump_char();
                    self.bump_char();
                    while let Some(character) = self.peek_char() {
                        if character == '\n' {
                            break;
                        }
                        self.bump_char();
                    }
                }
                _ => break,
            }
        }
    }

    fn lex_identifier(&mut self, span: Span) -> Token {
        let start = self.offset;
        while self
            .peek_char()
            .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            self.bump_char();
        }
        let text = &self.source[start..self.offset];
        let kind = match text {
            "fn" => TokenKind::Fn,
            "return" => TokenKind::Return,
            "u8" => TokenKind::U8,
            _ => TokenKind::Identifier(text.to_owned()),
        };
        Token { kind, span }
    }

    fn lex_number(&mut self, span: Span) -> Result<Token, Diagnostic> {
        let start = self.offset;
        if self.peek_char() == Some('0') && matches!(self.peek_second_char(), Some('x' | 'X')) {
            self.bump_char();
            self.bump_char();
            let digits_start = self.offset;
            while self
                .peek_char()
                .is_some_and(|character| character.is_ascii_hexdigit())
            {
                self.bump_char();
            }
            if digits_start == self.offset {
                return Err(Diagnostic::at(
                    span,
                    "hexadecimal literal must contain at least one digit",
                ));
            }
        } else {
            while self
                .peek_char()
                .is_some_and(|character| character.is_ascii_digit())
            {
                self.bump_char();
            }
        }
        Ok(Token {
            kind: TokenKind::Number(self.source[start..self.offset].to_owned()),
            span,
        })
    }

    fn lex(mut self) -> Result<Vec<Token>, Diagnostic> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            let span = self.current_span();
            let Some(character) = self.peek_char() else {
                tokens.push(Token {
                    kind: TokenKind::Eof,
                    span,
                });
                return Ok(tokens);
            };

            let token = if character.is_ascii_alphabetic() || character == '_' {
                self.lex_identifier(span)
            } else if character.is_ascii_digit() {
                self.lex_number(span)?
            } else if character == '-' && self.peek_second_char() == Some('>') {
                self.bump_char();
                self.bump_char();
                Token {
                    kind: TokenKind::Arrow,
                    span,
                }
            } else {
                self.bump_char();
                let kind = match character {
                    '(' => TokenKind::LeftParen,
                    ')' => TokenKind::RightParen,
                    '{' => TokenKind::LeftBrace,
                    '}' => TokenKind::RightBrace,
                    '+' => TokenKind::Plus,
                    ';' => TokenKind::Semicolon,
                    _ => {
                        return Err(Diagnostic::at(
                            span,
                            format!("unexpected character {character:?}"),
                        ));
                    }
                };
                Token { kind, span }
            };
            tokens.push(token);
        }
    }
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    fn current(&self) -> &Token {
        &self.tokens[self.position]
    }

    fn bump(&mut self) -> Token {
        let token = self.current().clone();
        if !matches!(token.kind, TokenKind::Eof) {
            self.position += 1;
        }
        token
    }

    fn syntax_error(&self, expected: &str) -> Diagnostic {
        Diagnostic::at(
            self.current().span,
            format!(
                "expected {expected}, found {}",
                self.current().kind.description()
            ),
        )
    }

    fn expect_simple(
        &mut self,
        expected: TokenKind,
        description: &str,
    ) -> Result<Token, Diagnostic> {
        if std::mem::discriminant(&self.current().kind) == std::mem::discriminant(&expected) {
            Ok(self.bump())
        } else {
            Err(self.syntax_error(description))
        }
    }

    fn expect_identifier(&mut self, expected: &str) -> Result<Token, Diagnostic> {
        match &self.current().kind {
            TokenKind::Identifier(name) if name == expected => Ok(self.bump()),
            TokenKind::Identifier(name) => Err(Diagnostic::at(
                self.current().span,
                format!("only function `{expected}` is supported, found `{name}`"),
            )),
            _ => Err(self.syntax_error(&format!("function name `{expected}`"))),
        }
    }

    fn parse_program(mut self) -> Result<Program, Diagnostic> {
        self.expect_simple(TokenKind::Fn, "`fn`")?;
        self.expect_identifier("main")?;
        self.expect_simple(TokenKind::LeftParen, "`(`")?;
        if !matches!(self.current().kind, TokenKind::RightParen) {
            return Err(Diagnostic::at(
                self.current().span,
                "Toy8 v1 `main` takes no parameters",
            ));
        }
        self.expect_simple(TokenKind::RightParen, "`)`")?;
        self.expect_simple(TokenKind::Arrow, "`->`")?;
        if !matches!(self.current().kind, TokenKind::U8) {
            return Err(Diagnostic::at(
                self.current().span,
                "Toy8 v1 supports only the return type `u8`",
            ));
        }
        self.bump();
        self.expect_simple(TokenKind::LeftBrace, "`{`")?;
        self.expect_simple(TokenKind::Return, "`return`")?;

        let mut literals = vec![self.parse_byte_literal()?];
        while matches!(self.current().kind, TokenKind::Plus) {
            self.bump();
            let literal = self.parse_byte_literal()?;
            literals.push(literal);
            if literals.len() > MAX_TERMS {
                return Err(Diagnostic::at(
                    literals[MAX_TERMS].span,
                    format!("sum exceeds the {MAX_TERMS}-literal limit for a 256-word Tiny8 ROM"),
                ));
            }
        }

        self.expect_simple(TokenKind::Semicolon, "`;` after the return expression")?;
        self.expect_simple(TokenKind::RightBrace, "`}`")?;
        if !matches!(self.current().kind, TokenKind::Eof) {
            return Err(self.syntax_error("end of file (only one function is supported)"));
        }

        Ok(Program { literals })
    }

    fn parse_byte_literal(&mut self) -> Result<ByteLiteral, Diagnostic> {
        let token = self.expect_simple(TokenKind::Number(String::new()), "a byte literal")?;
        let TokenKind::Number(text) = token.kind else {
            unreachable!("expect_simple checked the token kind")
        };
        let parsed = if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X"))
        {
            u128::from_str_radix(hex, 16)
        } else {
            text.parse::<u128>()
        };
        let value = parsed.map_err(|_| {
            Diagnostic::at(
                token.span,
                "integer literal is too large; Toy8 byte literals must be in 0..=255",
            )
        })?;
        if value > u8::MAX as u128 {
            return Err(Diagnostic::at(
                token.span,
                format!("u8 literal {text} is outside the range 0..=255"),
            ));
        }
        Ok(ByteLiteral {
            value: value as u8,
            span: token.span,
        })
    }
}

/// Lex and parse the deliberately restricted Toy8 v1 source language.
pub fn parse_source(source: &str) -> Result<Program, Diagnostic> {
    let tokens = Lexer::new(source).lex()?;
    Parser::new(tokens).parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_decimal_and_hex_literals_with_comments() {
        let program = parse_source(
            "// one-line comment\nfn main() -> u8 { return 2 + 0x03 + 4; } // trailing comment\n",
        )
        .unwrap();
        assert_eq!(
            program
                .literals
                .iter()
                .map(|lit| lit.value)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert_eq!(program.evaluate(), 9);
    }

    #[test]
    fn rejects_wrong_entry_point_parameters_and_return_type() {
        assert!(
            parse_source("fn other() -> u8 { return 1; }")
                .unwrap_err()
                .message
                .contains("only function `main`")
        );
        assert!(
            parse_source("fn main(x) -> u8 { return 1; }")
                .unwrap_err()
                .message
                .contains("takes no parameters")
        );
        assert!(
            parse_source("fn main() -> i8 { return 1; }")
                .unwrap_err()
                .message
                .contains("only the return type `u8`")
        );
    }

    #[test]
    fn rejects_out_of_range_and_malformed_literals() {
        assert!(
            parse_source("fn main() -> u8 { return 256; }")
                .unwrap_err()
                .message
                .contains("outside the range 0..=255")
        );
        assert!(
            parse_source("fn main() -> u8 { return 0x; }")
                .unwrap_err()
                .message
                .contains("at least one digit")
        );
        assert!(parse_source("fn main() -> u8 { return 1 +; }").is_err());
        assert!(parse_source("fn main() -> u8 { return (1 + 2); }").is_err());
    }

    #[test]
    fn rejects_trailing_programs_and_oversized_images() {
        assert!(
            parse_source("fn main() -> u8 { return 1; } fn main() -> u8 { return 2; }")
                .unwrap_err()
                .message
                .contains("only one function")
        );

        let sum = std::iter::repeat_n("1", MAX_TERMS + 1)
            .collect::<Vec<_>>()
            .join(" + ");
        let source = format!("fn main() -> u8 {{ return {sum}; }}");
        assert!(
            parse_source(&source)
                .unwrap_err()
                .message
                .contains("256-word Tiny8 ROM")
        );
    }

    #[test]
    fn evaluates_wrapping_addition_modulo_256() {
        let program = parse_source("fn main() -> u8 { return 255 + 1 + 2; }").unwrap();
        assert_eq!(program.evaluate(), 2);
    }
}
