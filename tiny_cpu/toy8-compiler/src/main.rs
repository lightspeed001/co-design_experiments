use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};
use toy8_compiler::compile_source;

const USAGE: &str = "Usage: toy8-compiler <source.toy8> [-o ROM.hex] [--mlir module.mlir] [--asm listing.asm]\n\nThe default outputs are named after the source file and use .hex, .mlir, and .asm extensions.";

#[derive(Debug)]
struct Options {
    source: PathBuf,
    rom: Option<PathBuf>,
    mlir: Option<PathBuf>,
    assembly: Option<PathBuf>,
}

fn parse_options() -> Result<Option<Options>, String> {
    let mut arguments = env::args().skip(1);
    let Some(first) = arguments.next() else {
        return Err(USAGE.to_owned());
    };
    if first == "--help" || first == "-h" {
        return Ok(None);
    }

    let mut options = Options {
        source: PathBuf::from(first),
        rom: None,
        mlir: None,
        assembly: None,
    };
    while let Some(argument) = arguments.next() {
        let destination = match argument.as_str() {
            "-o" | "--out" => &mut options.rom,
            "--mlir" => &mut options.mlir,
            "--asm" => &mut options.assembly,
            "--help" | "-h" => return Ok(None),
            _ => return Err(format!("unknown option `{argument}`\n\n{USAGE}")),
        };
        let path = arguments
            .next()
            .ok_or_else(|| format!("option `{argument}` needs a file path\n\n{USAGE}"))?;
        *destination = Some(PathBuf::from(path));
    }
    Ok(Some(options))
}

fn write_output(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(options) = parse_options().map_err(|message| std::io::Error::other(message))? else {
        println!("{USAGE}");
        return Ok(());
    };
    let source = fs::read_to_string(&options.source)?;
    let compilation = compile_source(&source)?;

    let rom_path = options
        .rom
        .unwrap_or_else(|| options.source.with_extension("hex"));
    let mlir_path = options
        .mlir
        .unwrap_or_else(|| options.source.with_extension("mlir"));
    let assembly_path = options
        .assembly
        .unwrap_or_else(|| options.source.with_extension("asm"));

    write_output(&rom_path, &compilation.hex)?;
    write_output(&mlir_path, &compilation.mlir)?;
    write_output(&assembly_path, &compilation.assembly)?;

    println!(
        "compiled {} byte literal(s) through verified MLIR to {} Tiny8 instructions",
        compilation.program.literals.len(),
        compilation.instructions.len()
    );
    println!(
        "static reference result: 0x{:02X}",
        compilation.expected_return
    );
    println!("MLIR: {}", mlir_path.display());
    println!("assembly listing: {}", assembly_path.display());
    println!("ROM image: {} (256 words)", rom_path.display());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("toy8-compiler: {error}");
            ExitCode::FAILURE
        }
    }
}
