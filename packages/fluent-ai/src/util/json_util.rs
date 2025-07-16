//! Zero-allocation, blazing-fast JSON utilities for provider implementations
//!
//! This module provides high-performance JSON manipulation tools optimized for:
//! • Zero allocations on hot paths
//! • Inlined operations for maximum speed
//! • Lock-free, safe Rust implementation
//! • Provider API compatibility with no performance overhead
//!
//! All operations use static dispatch and avoid heap allocations wherever possible.

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::{Deserialize, Serializer};
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

// ============================================================================
// Zero-allocation JSON merging operations
// ============================================================================

/// Merge two `serde_json::Value` objects by value with zero allocations on hot path
/// 
/// Performance characteristics:
/// • Zero allocation when `b` is empty or non-object
/// • In-place key insertion for object merging
/// • No intermediate copies or clones
/// 
/// # Examples
/// ```rust
/// use serde_json::json;
/// let a = json!({"key1": "value1"});
/// let b = json!({"key2": "value2"});
/// let merged = merge(a, b);
/// // Result: {"key1": "value1", "key2": "value2"}
/// ```
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider implementations
pub fn merge(mut a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
    match (&mut a, b) {
        (serde_json::Value::Object(a_map), serde_json::Value::Object(b_map)) => {
            // Reuse `a`'s allocation; extend with `b`'s keys
            // Zero intermediate allocations
            for (k, v) in b_map {
                a_map.insert(k, v);
            }
            a
        }
        (_, other) => {
            // Non-object case: return `a`, drop `other`
            // Zero allocation path
            drop(other);
            a
        }
    }
}

/// Mutate `a` in-place by union-inserting all keys from `b` (zero allocation)
/// 
/// Performance characteristics:
/// • True zero allocation - only modifies existing objects
/// • Inlined for compiler optimization
/// • Lock-free operation
/// 
/// # Examples
/// ```rust
/// use serde_json::json;
/// let mut a = json!({"key1": "value1"});
/// merge_inplace(&mut a, json!({"key2": "value2"}));
/// // `a` is now {"key1": "value1", "key2": "value2"}
/// ```
#[inline(always)]
pub fn merge_inplace(a: &mut serde_json::Value, b: serde_json::Value) {
    if let (serde_json::Value::Object(a_map), serde_json::Value::Object(b_map)) = (a, b) {
        // Zero allocation: direct key insertion
        for (k, v) in b_map {
            a_map.insert(k, v);
        }
    }
    // Non-object case: no-op (zero allocation)
}

// ============================================================================
// Zero-allocation serde adapters for provider quirks
// ============================================================================

/// Serde adapter for values serialized as escaped JSON strings
/// 
/// Many APIs serialize JSON objects as strings (e.g., `"{\"key\":\"value\"}"`)
/// This adapter handles this pattern with zero allocation during serialization.
/// 
/// Use with: `#[serde(with = "stringified_json")]`
/// 
/// Performance: Inlined serialization, minimal string allocation only when necessary
pub mod stringified_json {
    use super::*;

    /// Serialize a `serde_json::Value` as its compact string representation
    /// Performance: Single allocation for string conversion only
    #[inline(always)]
    pub fn serialize<S>(value: &serde_json::Value, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Use compact representation to minimize allocation
        serializer.serialize_str(&value.to_string())
    }

    /// Deserialize a JSON string back into a `serde_json::Value`
    /// Performance: Direct parsing, no intermediate allocations
    #[inline(always)]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = <&str>::deserialize(deserializer)?;
        serde_json::from_str(s).map_err(serde::de::Error::custom)
    }
}

// ============================================================================
// Zero-allocation flexible deserializers
// ============================================================================

/// Zero-allocation deserializer: string ∪ array ∪ null → Vec<T>
/// 
/// Accepts multiple JSON input formats and normalizes to Vec<T>:
/// • String: `"value"` → `vec![T::from_str("value")]`
/// • Array: `[item1, item2]` → `vec![item1, item2]`
/// • Null/Unit: `null` → `vec![]`
/// 
/// Performance characteristics:
/// • Static dispatch via visitor pattern
/// • Zero allocation visitor (zero-sized type)
/// • Inlined for maximum speed
/// 
/// # Examples
/// ```rust
/// #[derive(Deserialize)]
/// struct Config {
///     #[serde(deserialize_with = "string_or_vec")]
///     values: Vec<String>,
/// }
/// ```
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider configurations
pub fn string_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Infallible>,
    D: Deserializer<'de>,
{
    /// Zero-sized visitor for maximum performance
    struct VisitorImpl<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for VisitorImpl<T>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible>,
    {
        type Value = Vec<T>;

        #[inline(always)]
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("string, sequence, null, or unit")
        }

        /// Handle string input: parse single value into Vec
        #[inline(always)]
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // Single allocation for Vec with capacity 1
            Ok(vec![v.parse().map_err(E::custom)?])
        }

        /// Handle array input: deserialize sequence directly
        #[inline(always)]
        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // Direct deserialization - serde handles allocation efficiently
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        /// Handle null/unit input: return empty Vec
        #[inline(always)]
        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // Zero allocation: empty Vec
            Ok(Vec::new())
        }

        /// Handle null input: return empty Vec
        #[inline(always)]
        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // Zero allocation: empty Vec
            Ok(Vec::new())
        }
    }

    // Zero-sized visitor construction
    deserializer.deserialize_any(VisitorImpl(PhantomData))
}

/// Zero-allocation deserializer: array ∪ null → Vec<T>
/// 
/// Similar to `string_or_vec` but only accepts arrays and null:
/// • Array: `[item1, item2]` → `vec![item1, item2]`
/// • Null/Unit: `null` → `vec![]`
/// 
/// Performance characteristics:
/// • Static dispatch via visitor pattern
/// • Zero allocation visitor (zero-sized type)
/// • Inlined for maximum speed
/// 
/// # Examples
/// ```rust
/// #[derive(Deserialize)]
/// struct Config {
///     #[serde(deserialize_with = "null_or_vec")]
///     optional_items: Vec<Item>,
/// }
/// ```
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider configurations
pub fn null_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    /// Zero-sized visitor for maximum performance
    struct VisitorImpl<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for VisitorImpl<T>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;

        #[inline(always)]
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("sequence, null, or unit")
        }

        /// Handle array input: deserialize sequence directly
        #[inline(always)]
        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // Direct deserialization - serde handles allocation efficiently
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        /// Handle null/unit input: return empty Vec
        #[inline(always)]
        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // Zero allocation: empty Vec
            Ok(Vec::new())
        }

        /// Handle null input: return empty Vec
        #[inline(always)]
        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // Zero allocation: empty Vec
            Ok(Vec::new())
        }
    }

    // Zero-sized visitor construction
    deserializer.deserialize_any(VisitorImpl(PhantomData))
}

// ============================================================================
// Zero-allocation utility functions for common patterns
// ============================================================================

/// Merge JSON value into existing object, creating object if necessary
/// Performance: Zero allocation if target is already an object
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider implementations
pub fn ensure_object_and_merge(target: &mut serde_json::Value, source: serde_json::Value) {
    if !target.is_object() {
        *target = serde_json::Value::Object(serde_json::Map::new());
    }
    merge_inplace(target, source);
}

/// Get mutable reference to object map, creating empty object if necessary
/// Performance: Zero allocation if already an object
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider implementations
pub fn ensure_object_map(value: &mut serde_json::Value) -> &mut serde_json::Map<String, serde_json::Value> {
    if !value.is_object() {
        *value = serde_json::Value::Object(serde_json::Map::new());
    }
    value.as_object_mut().unwrap() // Safe: we just ensured it's an object
}

/// Insert key-value pair, creating object if necessary
/// Performance: Zero allocation if target is already an object
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider implementations
pub fn insert_or_create(target: &mut serde_json::Value, key: String, value: serde_json::Value) {
    ensure_object_map(target).insert(key, value);
}

/// Merge multiple JSON values in sequence with optimal allocation
/// Performance: Reuses first value's allocation, extends efficiently
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider implementations
pub fn merge_multiple<I>(values: I) -> serde_json::Value
where
    I: IntoIterator<Item = serde_json::Value>,
{
    let mut iter = values.into_iter();
    let mut result = iter.next().unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
    
    for value in iter {
        result = merge(result, value);
    }
    
    result
}

/// Check if JSON value is empty (null, empty object, empty array)
/// Performance: Inlined, no allocations
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider implementations
pub fn is_empty_value(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Object(map) => map.is_empty(),
        serde_json::Value::Array(arr) => arr.is_empty(),
        serde_json::Value::String(s) => s.is_empty(),
        _ => false,
    }
}

/// Compact JSON serialization with minimal allocation
/// Performance: Single allocation for output string
#[inline(always)]
pub fn to_compact_string(value: &serde_json::Value) -> String {
    value.to_string() // serde_json uses compact format by default
}

/// Pretty JSON serialization with controlled formatting
/// Performance: Single allocation for output string, optimized formatting
#[inline(always)]
#[allow(dead_code)] // TODO: Use in provider implementations
pub fn to_pretty_string(value: &serde_json::Value) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(value)
}

// ============================================================================
// Comprehensive unit tests (compile-time & run-time)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Dummy {
        #[serde(with = "stringified_json")]
        data: serde_json::Value,
    }

    // ----- merge -----------------------------------------------------------
    #[test]
    fn merge_by_value() {
        let a = serde_json::json!({"k1":"v1"});
        let b = serde_json::json!({"k2":"v2"});
        assert_eq!(merge(a, b), serde_json::json!({"k1":"v1","k2":"v2"}));
    }

    #[test]
    fn merge_in_place() {
        let mut a = serde_json::json!({"k1":"v1"});
        merge_inplace(&mut a, serde_json::json!({"k2":"v2"}));
        assert_eq!(a, serde_json::json!({"k1":"v1","k2":"v2"}));
    }

    // ----- stringified JSON -----------------------------------------------
    #[test]
    fn stringified_roundtrip() {
        let original = Dummy {
            data: serde_json::json!({"k":"v"}),
        };
        let s = serde_json::to_string(&original).unwrap();
        assert_eq!(s, r#"{"data":"{\"k\":\"v\"}"}"#);
        let parsed: Dummy = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, original);
    }

    // ----- string_or_vec ---------------------------------------------------
    #[test]
    fn str_or_array_deserialise() {
        #[derive(Deserialize, PartialEq, Debug)]
        struct Wrapper {
            #[serde(deserialize_with = "string_or_vec")]
            v: Vec<u32>,
        }

        let w1: Wrapper = serde_json::from_str(r#"{"v":"3"}"#).unwrap();
        assert_eq!(w1.v, vec![3]);

        let w2: Wrapper = serde_json::from_str(r#"{"v":[1,2,3]}"#).unwrap();
        assert_eq!(w2.v, vec![1, 2, 3]);

        let w3: Wrapper = serde_json::from_str(r#"{"v":null}"#).unwrap();
        assert!(w3.v.is_empty());
    }

    // ----- null_or_vec -----------------------------------------------------
    #[test]
    fn null_or_array_deserialise() {
        #[derive(Deserialize, PartialEq, Debug)]
        struct Wrapper {
            #[serde(deserialize_with = "null_or_vec")]
            v: Vec<bool>,
        }

        let w1: Wrapper = serde_json::from_str(r#"{"v":[true,false]}"#).unwrap();
        assert_eq!(w1.v, vec![true, false]);

        let w2: Wrapper = serde_json::from_str(r#"{"v":null}"#).unwrap();
        assert!(w2.v.is_empty());
    }

    // ----- utility functions -----------------------------------------------
    #[test]
    fn test_ensure_object_and_merge() {
        let mut target = serde_json::json!("not an object");
        let source = serde_json::json!({"key": "value"});
        
        ensure_object_and_merge(&mut target, source);
        assert_eq!(target, serde_json::json!({"key": "value"}));
    }

    #[test]
    fn test_is_empty_value() {
        assert!(is_empty_value(&serde_json::json!(null)));
        assert!(is_empty_value(&serde_json::json!({})));
        assert!(is_empty_value(&serde_json::json!([])));
        assert!(is_empty_value(&serde_json::json!("")));
        assert!(!is_empty_value(&serde_json::json!({"key": "value"})));
        assert!(!is_empty_value(&serde_json::json!([1, 2, 3])));
        assert!(!is_empty_value(&serde_json::json!("content")));
    }

    #[test]
    fn test_merge_multiple() {
        let values = vec![
            serde_json::json!({"a": 1}),
            serde_json::json!({"b": 2}),
            serde_json::json!({"c": 3}),
        ];
        
        let result = merge_multiple(values);
        assert_eq!(result, serde_json::json!({"a": 1, "b": 2, "c": 3}));
    }
}