use std::{error::Error, fmt};

/// One-based source position used in lexer and parser diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub const fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }
}

/// A user-facing compiler diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub span: Option<Span>,
    pub message: String,
}

impl Diagnostic {
    pub fn at(span: Span, message: impl Into<String>) -> Self {
        Self {
            span: Some(span),
            message: message.into(),
        }
    }

    pub fn global(message: impl Into<String>) -> Self {
        Self {
            span: None,
            message: message.into(),
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(span) = self.span {
            write!(
                formatter,
                "{}:{}: error: {}",
                span.line, span.column, self.message
            )
        } else {
            write!(formatter, "error: {}", self.message)
        }
    }
}

impl Error for Diagnostic {}
