//! Core filter evaluation logic
//!
//! Handles the main evaluation entry points for filter expressions including
//! predicate evaluation and expression evaluation with context support.

use std::collections::HashSet;

use super::comparison::ValueComparator;
use super::property::PropertyResolver;
use super::selectors::SelectorEvaluator;
use super::utils::FilterUtils;
use crate::json_path::error::{JsonPathResult, deserialization_error};
use crate::json_path::functions::FunctionEvaluator;
use crate::json_path::parser::{ComparisonOp, FilterExpression, FilterValue, LogicalOp};

/// Filter Expression Evaluator
pub struct FilterEvaluator;

impl FilterEvaluator {
    /// Evaluate filter predicate against JSON context
    #[inline]
    pub fn evaluate_predicate(
        context: &serde_json::Value,
        expr: &FilterExpression,
    ) -> JsonPathResult<bool> {
        // Use empty context for backward compatibility
        let empty_context = HashSet::new();
        Self::evaluate_predicate_with_context(context, expr, &empty_context)
    }

    /// Evaluate filter predicate with property existence context
    #[inline]
    pub fn evaluate_predicate_with_context(
        context: &serde_json::Value,
        expr: &FilterExpression,
        existing_properties: &HashSet<String>,
    ) -> JsonPathResult<bool> {
        println!(
            "DEBUG: evaluate_predicate called with context={:?}, expr={:?}",
            serde_json::to_string(context).unwrap_or("invalid".to_string()),
            expr
        );
        match expr {
            FilterExpression::Property { path } => {
                // RFC 9535: Property access in filter context checks existence and truthiness
                // For @.author, this should return false if the object doesn't have an 'author' property
                println!("DEBUG: Evaluating property filter with path={:?}", path);
                PropertyResolver::property_exists_and_truthy(context, path)
            }
            FilterExpression::Comparison {
                left,
                operator,
                right,
            } => {
                let left_val =
                    Self::evaluate_expression_with_context(context, left, existing_properties)?;
                let right_val =
                    Self::evaluate_expression_with_context(context, right, existing_properties)?;
                ValueComparator::compare_values_with_context(
                    &left_val,
                    *operator,
                    &right_val,
                    existing_properties,
                )
            }
            FilterExpression::Logical {
                left,
                operator,
                right,
            } => {
                let left_result =
                    Self::evaluate_predicate_with_context(context, left, existing_properties)?;
                let right_result =
                    Self::evaluate_predicate_with_context(context, right, existing_properties)?;
                Ok(match operator {
                    LogicalOp::And => left_result && right_result,
                    LogicalOp::Or => left_result || right_result,
                })
            }
            FilterExpression::Function { name, args } => {
                let value = FunctionEvaluator::evaluate_function_value(
                    context,
                    name,
                    args,
                    &|ctx, expr| {
                        Self::evaluate_expression_with_context(ctx, expr, existing_properties)
                    },
                )?;
                Ok(FilterUtils::is_truthy(&value))
            }
            _ => Ok(FilterUtils::is_truthy(
                &Self::evaluate_expression_with_context(context, expr, existing_properties)?,
            )),
        }
    }

    /// Evaluate expression to get its value
    #[inline]
    pub fn evaluate_expression(
        context: &serde_json::Value,
        expr: &FilterExpression,
    ) -> JsonPathResult<FilterValue> {
        let empty_context = HashSet::new();
        Self::evaluate_expression_with_context(context, expr, &empty_context)
    }

    /// Evaluate expression with property context
    #[inline]
    pub fn evaluate_expression_with_context(
        context: &serde_json::Value,
        expr: &FilterExpression,
        existing_properties: &HashSet<String>,
    ) -> JsonPathResult<FilterValue> {
        match expr {
            FilterExpression::Current => Ok(PropertyResolver::json_value_to_filter_value(context)),
            FilterExpression::Property { path } => {
                PropertyResolver::resolve_property_path_with_context(
                    context,
                    path,
                    existing_properties,
                )
            }
            FilterExpression::JsonPath { selectors } => {
                SelectorEvaluator::evaluate_jsonpath_selectors(context, selectors)
            }
            FilterExpression::Literal { value } => Ok(value.clone()),
            FilterExpression::Function { name, args } => {
                FunctionEvaluator::evaluate_function_value(context, name, args, &|ctx, expr| {
                    Self::evaluate_expression_with_context(ctx, expr, existing_properties)
                })
            }
            _ => Err(deserialization_error(
                "complex expressions not supported in value context".to_string(),
                format!("{:?}", expr),
                "FilterValue",
            )),
        }
    }
}
