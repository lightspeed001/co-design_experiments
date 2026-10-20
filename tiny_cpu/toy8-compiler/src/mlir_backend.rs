use crate::{diagnostic::Diagnostic, frontend::Program, target::Instruction};
use melior::{
    Context,
    dialect::{DialectRegistry, arith, func},
    ir::{
        Attribute, AttributeLike, Block, Location, Module, Region, RegionLike, Type, Value,
        ValueLike,
        attribute::{StringAttribute, TypeAttribute},
        block::BlockLike,
        operation::OperationLike,
        r#type::FunctionType,
    },
    utility::register_all_dialects,
};
use mlir_sys::{mlirAttributeIsAInteger, mlirIntegerAttrGetValueUInt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueState {
    Constant(u8),
    Accumulator,
}

/// Build and verify a Melior module, then select Tiny8 instructions from the
/// verified MLIR operation tree. The returned text is the exact verified IR.
pub fn compile_mlir_to_tiny8(program: &Program) -> Result<(String, Vec<Instruction>), Diagnostic> {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let module = build_module(&context, program)?;
    if !module.as_operation().verify() {
        return Err(Diagnostic::global(
            "MLIR verifier rejected the generated Toy8 module",
        ));
    }

    let mlir_text = module.as_operation().to_string();
    let instructions = select_tiny8_instructions(&module)?;
    Ok((mlir_text, instructions))
}

fn build_module<'c>(context: &'c Context, program: &Program) -> Result<Module<'c>, Diagnostic> {
    let location = Location::unknown(context);
    let module = Module::new(location);
    let byte_type = Type::parse(context, "i8")
        .ok_or_else(|| Diagnostic::global("internal error: could not parse MLIR type i8"))?;

    let block = Block::new(&[]);
    let mut previous_value: Option<Value<'c, '_>> = None;

    for literal in &program.literals {
        let attribute = Attribute::parse(context, &format!("{} : i8", literal.value))
            .ok_or_else(|| Diagnostic::global("internal error: could not build i8 constant"))?;
        let constant = block.append_operation(arith::constant(context, attribute, location));
        let value: Value<'c, '_> = constant
            .result(0)
            .map_err(|error| Diagnostic::global(format!("internal MLIR constant error: {error}")))?
            .into();

        previous_value = Some(if let Some(lhs) = previous_value {
            let sum = block.append_operation(arith::addi(lhs, value, location));
            sum.result(0)
                .map_err(|error| Diagnostic::global(format!("internal MLIR add error: {error}")))?
                .into()
        } else {
            value
        });
    }

    let result = previous_value
        .ok_or_else(|| Diagnostic::global("internal error: parsed Toy8 program has no literals"))?;
    block.append_operation(func::r#return(&[result], location));

    let region = Region::new();
    region.append_block(block);
    let function_type = FunctionType::new(context, &[], &[byte_type]);
    let function = func::func(
        context,
        StringAttribute::new(context, "main"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );
    module.body().append_operation(function);
    Ok(module)
}

fn select_tiny8_instructions(module: &Module<'_>) -> Result<Vec<Instruction>, Diagnostic> {
    let function = module
        .body()
        .first_operation()
        .ok_or_else(|| Diagnostic::global("backend expected one func.func operation"))?;
    let function_name = function
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-name>")
        .to_owned();
    if function_name != "func.func" || function.region_count() != 1 {
        return Err(Diagnostic::global(format!(
            "unsupported top-level MLIR operation `{function_name}`"
        )));
    }

    let block = function
        .region(0)
        .map_err(|error| Diagnostic::global(format!("invalid function region: {error}")))?
        .first_block()
        .ok_or_else(|| Diagnostic::global("backend expected an entry block"))?;
    if block.argument_count() != 0 {
        return Err(Diagnostic::global(
            "Tiny8 v1 backend does not support function arguments",
        ));
    }

    let mut bindings: Vec<(Value<'_, '_>, ValueState)> = Vec::new();
    let mut instructions = Vec::new();
    let mut cursor = block.first_operation();
    let mut saw_return = false;

    while let Some(operation) = cursor {
        let next = operation.next_in_block();
        let name = operation
            .name()
            .as_string_ref()
            .as_str()
            .unwrap_or("<invalid-name>")
            .to_owned();

        match name.as_str() {
            "arith.constant" => {
                if operation.result_count() != 1 || operation.operand_count() != 0 {
                    return Err(Diagnostic::global(
                        "backend expected arith.constant to have one result and no operands",
                    ));
                }
                let result = operation.result(0).map_err(|error| {
                    Diagnostic::global(format!("invalid constant result: {error}"))
                })?;
                if result.r#type().to_string() != "i8" {
                    return Err(Diagnostic::global("backend supports only i8 constants"));
                }
                let value_attribute = operation.attribute("value").map_err(|error| {
                    Diagnostic::global(format!("constant lacks value attribute: {error}"))
                })?;
                let raw_attribute = value_attribute.to_raw();
                if !unsafe { mlirAttributeIsAInteger(raw_attribute) } {
                    return Err(Diagnostic::global("backend expected an integer constant"));
                }
                let value = unsafe { mlirIntegerAttrGetValueUInt(raw_attribute) };
                let value = u8::try_from(value).map_err(|_| {
                    Diagnostic::global("backend encountered an integer constant outside u8")
                })?;
                bindings.push((Value::from(result), ValueState::Constant(value)));
            }
            "arith.addi" => {
                if operation.operand_count() != 2 || operation.result_count() != 1 {
                    return Err(Diagnostic::global(
                        "backend expected arith.addi to have two operands and one result",
                    ));
                }
                let lhs = operation
                    .operand(0)
                    .map_err(|error| Diagnostic::global(format!("invalid add lhs: {error}")))?;
                let rhs = operation
                    .operand(1)
                    .map_err(|error| Diagnostic::global(format!("invalid add rhs: {error}")))?;
                let lhs_state = find_value_state(&bindings, lhs).ok_or_else(|| {
                    Diagnostic::global("backend could not trace the left arith.addi operand")
                })?;
                let rhs_state = find_value_state(&bindings, rhs).ok_or_else(|| {
                    Diagnostic::global("backend could not trace the right arith.addi operand")
                })?;
                let ValueState::Constant(rhs_value) = rhs_state else {
                    return Err(Diagnostic::global(
                        "Tiny8 v1 supports arith.addi only when its right operand is a constant",
                    ));
                };
                match lhs_state {
                    ValueState::Constant(lhs_value) => {
                        instructions.push(Instruction::Ldi(lhs_value))
                    }
                    ValueState::Accumulator => {}
                }
                instructions.push(Instruction::Addi(rhs_value));

                let result = operation
                    .result(0)
                    .map_err(|error| Diagnostic::global(format!("invalid add result: {error}")))?;
                if result.r#type().to_string() != "i8" {
                    return Err(Diagnostic::global("backend supports only i8 addition"));
                }
                bindings.push((Value::from(result), ValueState::Accumulator));
            }
            "func.return" => {
                if operation.operand_count() != 1 || operation.result_count() != 0 {
                    return Err(Diagnostic::global(
                        "Tiny8 v1 expects func.return to return exactly one value",
                    ));
                }
                let returned_value = operation.operand(0).map_err(|error| {
                    Diagnostic::global(format!("invalid return operand: {error}"))
                })?;
                match find_value_state(&bindings, returned_value).ok_or_else(|| {
                    Diagnostic::global("backend could not trace the returned MLIR value")
                })? {
                    ValueState::Constant(value) => instructions.push(Instruction::Ldi(value)),
                    ValueState::Accumulator => {}
                }
                instructions.push(Instruction::Sta(0x10));
                instructions.push(Instruction::Halt);
                if next.is_some() {
                    return Err(Diagnostic::global(
                        "backend does not support operations after func.return",
                    ));
                }
                saw_return = true;
            }
            _ => {
                return Err(Diagnostic::global(format!(
                    "unsupported MLIR operation `{name}` in Tiny8 v1 backend"
                )));
            }
        }

        cursor = next;
    }

    if !saw_return {
        return Err(Diagnostic::global(
            "backend expected a func.return operation",
        ));
    }
    Ok(instructions)
}

fn find_value_state(
    bindings: &[(Value<'_, '_>, ValueState)],
    value: Value<'_, '_>,
) -> Option<ValueState> {
    bindings
        .iter()
        .find_map(|(known_value, state)| (*known_value == value).then_some(*state))
}
