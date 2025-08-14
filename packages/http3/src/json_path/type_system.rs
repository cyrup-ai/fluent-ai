//! RFC 9535 JSONPath Function Type System (Section 2.4.1-2.4.3)
//!
//! Implements the type system for function expressions including:
//! - ValueType: The type of any JSON value
//! - LogicalType: The type of test or logical expression results (true/false)
//! - NodesType: The type of a nodelist
//! - Type conversion rules
//! - Well-typedness validation for function expressions

use crate::json_path::{
    ast::{FilterExpression, FilterValue},
    error::{JsonPathResult, invalid_expression_error},
};

/// RFC 9535 Function Expression Type System
///
/// Defines the three core types used in JSONPath function expressions
/// and provides type checking and conversion capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionType {
    /// ValueType: The type of any JSON value
    /// Can represent strings, numbers, booleans, null, arrays, or objects
    ValueType,

    /// LogicalType: The type of test or logical expression results
    /// Represents boolean true/false values from comparisons and logical operations
    LogicalType,

    /// NodesType: The type of a nodelist
    /// Represents the result of JSONPath expressions that select multiple nodes
    NodesType,
}

/// Type-safe wrapper for function expression values
///
/// Provides compile-time type safety and runtime type checking
/// for function arguments and return values.
#[derive(Debug, Clone)]
pub enum TypedValue {
    /// A JSON value with ValueType
    Value(serde_json::Value),

    /// A boolean result with LogicalType
    Logical(bool),

    /// A nodelist with NodesType
    Nodes(Vec<serde_json::Value>),
}

/// Function type signature definition
///
/// Defines the expected parameter types and return type for a function.
/// Used for compile-time type checking of function expressions.
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Expected parameter types in order
    pub parameter_types: Vec<FunctionType>,
    /// Return type of the function
    pub return_type: FunctionType,
    /// Function name for error reporting
    pub name: String,
}

/// RFC 9535 Function Type System Implementation
pub struct TypeSystem;

impl TypeSystem {
    /// Get function signature for built-in RFC 9535 functions
    ///
    /// Returns the type signature for the specified function name,
    /// or None if the function is not a built-in RFC 9535 function.
    #[inline]
    pub fn get_function_signature(function_name: &str) -> Option<FunctionSignature> {
        match function_name {
            "length" => Some(FunctionSignature {
                parameter_types: vec![FunctionType::ValueType],
                return_type: FunctionType::ValueType,
                name: "length".to_string(),
            }),
            "count" => Some(FunctionSignature {
                parameter_types: vec![FunctionType::NodesType],
                return_type: FunctionType::ValueType,
                name: "count".to_string(),
            }),
            "match" => Some(FunctionSignature {
                parameter_types: vec![FunctionType::ValueType, FunctionType::ValueType],
                return_type: FunctionType::LogicalType,
                name: "match".to_string(),
            }),
            "search" => Some(FunctionSignature {
                parameter_types: vec![FunctionType::ValueType, FunctionType::ValueType],
                return_type: FunctionType::LogicalType,
                name: "search".to_string(),
            }),
            "value" => Some(FunctionSignature {
                parameter_types: vec![FunctionType::NodesType],
                return_type: FunctionType::ValueType,
                name: "value".to_string(),
            }),
            _ => None,
        }
    }

    /// RFC 9535 Section 2.4.2: Type Conversion Rules
    ///
    /// Performs type conversion according to RFC 9535 specifications:
    /// - ValueType can be converted to LogicalType using test expression conversion
    /// - NodesType can be converted to ValueType if nodelist has exactly one node
    #[inline]
    pub fn convert_type(
        value: TypedValue,
        target_type: FunctionType,
    ) -> JsonPathResult<TypedValue> {
        match (value, target_type) {
            // ValueType to LogicalType conversion (test expression conversion)
            (TypedValue::Value(json_val), FunctionType::LogicalType) => {
                let logical_result = Self::value_to_logical(&json_val);
                Ok(TypedValue::Logical(logical_result))
            }

            // NodesType to ValueType conversion (single node requirement)
            (TypedValue::Nodes(nodes), FunctionType::ValueType) => {
                if nodes.len() == 1 {
                    // Safe to access index 0 since we verified length == 1
                    let mut node_iter = nodes.into_iter();
                    match node_iter.next() {
                        Some(node) => Ok(TypedValue::Value(node)),
                        None => {
                            // This should never happen since len() == 1, but handle gracefully
                            Err(invalid_expression_error(
                                "",
                                "internal error: expected single node but iterator was empty",
                                None,
                            ))
                        }
                    }
                } else {
                    Err(invalid_expression_error(
                        "",
                        &format!(
                            "NodesType to ValueType conversion requires exactly one node, found {}",
                            nodes.len()
                        ),
                        None,
                    ))
                }
            }

            // Same type conversions (no-op)
            (TypedValue::Value(val), FunctionType::ValueType) => Ok(TypedValue::Value(val)),
            (TypedValue::Logical(val), FunctionType::LogicalType) => Ok(TypedValue::Logical(val)),
            (TypedValue::Nodes(val), FunctionType::NodesType) => Ok(TypedValue::Nodes(val)),

            // Invalid conversions
            (TypedValue::Logical(_), FunctionType::ValueType) => Err(invalid_expression_error(
                "",
                "LogicalType cannot be converted to ValueType",
                None,
            )),
            (TypedValue::Logical(_), FunctionType::NodesType) => Err(invalid_expression_error(
                "",
                "LogicalType cannot be converted to NodesType",
                None,
            )),
            (TypedValue::Value(_), FunctionType::NodesType) => Err(invalid_expression_error(
                "",
                "ValueType cannot be converted to NodesType",
                None,
            )),
            (TypedValue::Nodes(_), FunctionType::LogicalType) => Err(invalid_expression_error(
                "",
                "NodesType cannot be converted to LogicalType",
                None,
            )),
        }
    }

    /// RFC 9535 Section 2.4.3: Well-Typedness Validation
    ///
    /// Validates that a function expression is well-typed according to RFC rules:
    /// 1. The function is known (defined in RFC 9535 or registered extension)
    /// 2. The function is applied to the correct number of arguments
    /// 3. All function arguments are well-typed
    /// 4. All function arguments can be converted to declared parameter types
    #[inline]
    pub fn validate_function_expression(
        function_name: &str,
        arguments: &[FilterExpression],
    ) -> JsonPathResult<FunctionSignature> {
        // 1. Check if function is known
        let signature = Self::get_function_signature(function_name).ok_or_else(|| {
            invalid_expression_error("", &format!("unknown function: {}", function_name), None)
        })?;

        // 2. Check argument count
        if arguments.len() != signature.parameter_types.len() {
            return Err(invalid_expression_error(
                "",
                &format!(
                    "function {} expects {} arguments, got {}",
                    function_name,
                    signature.parameter_types.len(),
                    arguments.len()
                ),
                None,
            ));
        }

        // 3. Validate each argument is well-typed (recursive validation)
        for (i, arg) in arguments.iter().enumerate() {
            let expected_type = &signature.parameter_types[i];
            Self::validate_expression_type(arg, expected_type)?;
        }

        Ok(signature)
    }

    /// Validate that an expression produces the expected type
    ///
    /// Performs static type analysis on filter expressions to ensure
    /// they can produce values of the expected type.
    #[inline]
    fn validate_expression_type(
        expr: &FilterExpression,
        expected_type: &FunctionType,
    ) -> JsonPathResult<()> {
        let actual_type = Self::infer_expression_type(expr)?;

        // Check if types match exactly or can be converted
        if actual_type == *expected_type {
            return Ok(());
        }

        // Check if conversion is possible
        match (&actual_type, expected_type) {
            // ValueType can be converted to LogicalType
            (FunctionType::ValueType, FunctionType::LogicalType) => Ok(()),
            // NodesType can be converted to ValueType (runtime check needed)
            (FunctionType::NodesType, FunctionType::ValueType) => Ok(()),
            _ => Err(invalid_expression_error(
                "",
                &format!(
                    "type mismatch: expected {:?}, found {:?}",
                    expected_type, actual_type
                ),
                None,
            )),
        }
    }

    /// Infer the type that an expression will produce
    ///
    /// Performs static type inference on filter expressions to determine
    /// their return type without executing them.
    #[inline]
    fn infer_expression_type(expr: &FilterExpression) -> JsonPathResult<FunctionType> {
        match expr {
            FilterExpression::Current => Ok(FunctionType::ValueType),
            FilterExpression::Property { .. } => Ok(FunctionType::ValueType),
            FilterExpression::JsonPath { .. } => Ok(FunctionType::NodesType),
            FilterExpression::Literal { value } => match value {
                FilterValue::Boolean(_) => Ok(FunctionType::LogicalType),
                _ => Ok(FunctionType::ValueType),
            },
            FilterExpression::Comparison { .. } => Ok(FunctionType::LogicalType),
            FilterExpression::Logical { .. } => Ok(FunctionType::LogicalType),
            FilterExpression::Regex { .. } => Ok(FunctionType::LogicalType),
            FilterExpression::Function { name, args } => {
                // Validate the function expression and get its return type
                let signature = Self::validate_function_expression(name, args)?;
                Ok(signature.return_type)
            }
        }
    }

    /// Convert JSON value to logical type using test expression conversion
    ///
    /// RFC 9535: ValueType to LogicalType conversion uses the "truthiness" rules:
    /// - false and null are false
    /// - Numbers: zero is false, all others are true  
    /// - Strings: empty string is false, all others are true
    /// - Arrays and objects: always true (even if empty)
    #[inline]
    fn value_to_logical(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Null => false,
            serde_json::Value::Bool(b) => *b,
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i != 0
                } else if let Some(f) = n.as_f64() {
                    f != 0.0 && !f.is_nan()
                } else {
                    false
                }
            }
            serde_json::Value::String(s) => !s.is_empty(),
            serde_json::Value::Array(_) => true, // Always true, even if empty
            serde_json::Value::Object(_) => true, // Always true, even if empty
        }
    }

    /// Convert FilterValue to TypedValue
    ///
    /// Bridges the gap between FilterValue (used in existing code)
    /// and TypedValue (used in the new type system).
    #[inline]
    pub fn filter_value_to_typed_value(value: &FilterValue) -> TypedValue {
        match value {
            FilterValue::String(s) => TypedValue::Value(serde_json::Value::String(s.clone())),
            FilterValue::Number(n) => TypedValue::Value(serde_json::json!(*n)),
            FilterValue::Integer(i) => TypedValue::Value(serde_json::json!(*i)),
            FilterValue::Boolean(b) => TypedValue::Logical(*b),
            FilterValue::Null => TypedValue::Value(serde_json::Value::Null),
            FilterValue::Missing => TypedValue::Value(serde_json::Value::Null), /* Missing converts to null */
        }
    }

    /// Convert TypedValue to FilterValue
    ///
    /// Converts from the new type system back to the existing FilterValue
    /// for compatibility with existing code.
    #[inline]
    pub fn typed_value_to_filter_value(value: &TypedValue) -> JsonPathResult<FilterValue> {
        match value {
            TypedValue::Value(json_val) => match json_val {
                serde_json::Value::Null => Ok(FilterValue::Null),
                serde_json::Value::Bool(b) => Ok(FilterValue::Boolean(*b)),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(FilterValue::Integer(i))
                    } else if let Some(f) = n.as_f64() {
                        Ok(FilterValue::Number(f))
                    } else {
                        Ok(FilterValue::Null)
                    }
                }
                serde_json::Value::String(s) => Ok(FilterValue::String(s.clone())),
                _ => Err(invalid_expression_error(
                    "",
                    "arrays and objects cannot be converted to FilterValue",
                    None,
                )),
            },
            TypedValue::Logical(b) => Ok(FilterValue::Boolean(*b)),
            TypedValue::Nodes(_) => Err(invalid_expression_error(
                "",
                "NodesType cannot be converted to FilterValue",
                None,
            )),
        }
    }

    /// Create a nodelist TypedValue from a vector of JSON values
    ///
    /// Helper function for creating NodesType values from JSONPath evaluation results.
    #[inline]
    pub fn create_nodes_value(nodes: Vec<serde_json::Value>) -> TypedValue {
        TypedValue::Nodes(nodes)
    }

    /// Extract nodes from a TypedValue
    ///
    /// Returns the underlying node vector if the value is NodesType,
    /// otherwise returns an error.
    #[inline]
    pub fn extract_nodes(value: &TypedValue) -> JsonPathResult<&[serde_json::Value]> {
        match value {
            TypedValue::Nodes(nodes) => Ok(nodes),
            _ => Err(invalid_expression_error(
                "",
                "expected NodesType value",
                None,
            )),
        }
    }

    /// Check if a TypedValue is empty (for NodesType) or falsy (for other types)
    ///
    /// Used for optimizing filter expressions and short-circuit evaluation.
    #[inline]
    pub fn is_empty_or_falsy(value: &TypedValue) -> bool {
        match value {
            TypedValue::Value(json_val) => !Self::value_to_logical(json_val),
            TypedValue::Logical(b) => !*b,
            TypedValue::Nodes(nodes) => nodes.is_empty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_signatures() {
        let length_sig = TypeSystem::get_function_signature("length")
            .expect("Failed to get function signature for 'length'");
        assert_eq!(length_sig.parameter_types.len(), 1);
        assert_eq!(length_sig.parameter_types[0], FunctionType::ValueType);
        assert_eq!(length_sig.return_type, FunctionType::ValueType);

        let count_sig = TypeSystem::get_function_signature("count")
            .expect("Failed to get function signature for 'count'");
        assert_eq!(count_sig.parameter_types.len(), 1);
        assert_eq!(count_sig.parameter_types[0], FunctionType::NodesType);
        assert_eq!(count_sig.return_type, FunctionType::ValueType);

        assert!(TypeSystem::get_function_signature("unknown").is_none());
    }

    #[test]
    fn test_type_conversions() {
        // ValueType to LogicalType
        let string_val = TypedValue::Value(serde_json::json!("hello"));
        let logical = TypeSystem::convert_type(string_val, FunctionType::LogicalType)
            .expect("Failed to convert string ValueType to LogicalType");
        assert!(matches!(logical, TypedValue::Logical(true)));

        let empty_string_val = TypedValue::Value(serde_json::json!(""));
        let logical = TypeSystem::convert_type(empty_string_val, FunctionType::LogicalType)
            .expect("Failed to convert empty string ValueType to LogicalType");
        assert!(matches!(logical, TypedValue::Logical(false)));

        // NodesType to ValueType (single node)
        let single_node = TypedValue::Nodes(vec![serde_json::json!("value")]);
        let value = TypeSystem::convert_type(single_node, FunctionType::ValueType)
            .expect("Failed to convert single node NodesType to ValueType");
        assert!(matches!(value, TypedValue::Value(_)));

        // NodesType to ValueType (multiple nodes - should fail)
        let multi_nodes = TypedValue::Nodes(vec![serde_json::json!("a"), serde_json::json!("b")]);
        assert!(TypeSystem::convert_type(multi_nodes, FunctionType::ValueType).is_err());
    }

    #[test]
    fn test_value_to_logical() {
        assert!(!TypeSystem::value_to_logical(&serde_json::Value::Null));
        assert!(!TypeSystem::value_to_logical(&serde_json::json!(false)));
        assert!(TypeSystem::value_to_logical(&serde_json::json!(true)));
        assert!(!TypeSystem::value_to_logical(&serde_json::json!(0)));
        assert!(TypeSystem::value_to_logical(&serde_json::json!(1)));
        assert!(!TypeSystem::value_to_logical(&serde_json::json!("")));
        assert!(TypeSystem::value_to_logical(&serde_json::json!("hello")));
        assert!(TypeSystem::value_to_logical(&serde_json::json!([])));
        assert!(TypeSystem::value_to_logical(&serde_json::json!({})));
    }
}
