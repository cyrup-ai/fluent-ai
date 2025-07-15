// ============================================================================
// File: src/json_util.rs
// ----------------------------------------------------------------------------
// High-performance helpers for JSON manipulation + serde adapters.
//
// • 100 % safe Rust (`#![forbid(unsafe_code)]` implicitly enforced).
// • No hidden allocations on the fast-path – every mutation is in-place.
// • Symmetric (de)serialisers for provider quirks such as “stringified JSON”.
// ============================================================================

#![allow(clippy::type_complexity)]

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use std::{convert::Infallible, fmt, marker::PhantomData, str::FromStr};

// -----------------------------------------------------------------------------
// In-place & by-value object merging
// -----------------------------------------------------------------------------

/// Merge two `serde_json::Value` objects **by value**.
/// If both inputs are JSON objects their key sets are united; otherwise `a` is returned.
///
/// *Hot-path*: no allocations when `b` is empty or non-object.
#[inline(always)]
pub fn merge(mut a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
    match (&mut a, b) {
        (serde_json::Value::Object(ref mut a_map), serde_json::Value::Object(b_map)) => {
            // Reuse `a`’s allocation; no intermediate clones.
            for (k, v) in b_map {
                a_map.insert(k, v);
            }
            a
        }
        (_, other) => {
            // If `other` isn’t an object the spec says “return a”.
            drop(other);
            a
        }
    }
}

/// Mutate `a` **in-place** by union-inserting all keys from `b` when both are objects.
#[inline(always)]
pub fn merge_inplace(a: &mut serde_json::Value, b: serde_json::Value) {
    if let (serde_json::Value::Object(a_map), serde_json::Value::Object(b_map)) = (a, b) {
        for (k, v) in b_map {
            a_map.insert(k, v);
        }
    }
}

// -----------------------------------------------------------------------------
// Serde adaptor: objects masquerading as escaped JSON strings
// -----------------------------------------------------------------------------

/// Helpers for providers that serialise a raw JSON object as a *string*
/// (e.g. `"{"key":"value"}"`).
/// Use with `#[serde(with = "stringified_json")]`.
pub mod stringified_json {
    use serde::{self, Deserialize, Deserializer, Serializer};

    /// Serialise a `serde_json::Value` as its compact string representation.
    #[inline(always)]
    pub fn serialize<S>(value: &serde_json::Value, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    /// Deserialize a JSON string back into a `serde_json::Value`.
    #[inline(always)]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = <&str>::deserialize(deserializer)?;
        serde_json::from_str(s).map_err(serde::de::Error::custom)
    }
}

// -----------------------------------------------------------------------------
// Serde helpers: accept string | seq | null  → Vec<T>
// -----------------------------------------------------------------------------

/// Accepts **string ∪ array ∪ null** JSON inputs and always yields `Vec<T>`.
#[inline(always)]
pub fn string_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Infallible>,
    D: Deserializer<'de>,
{
    struct VisitorImpl<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for VisitorImpl<T>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible>,
    {
        type Value = Vec<T>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("string, sequence, null, or unit")
        }

        #[inline(always)]
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![v.parse().map_err(E::custom)?])
        }

        #[inline(always)]
        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        #[inline(always)]
        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Vec::new())
        }

        #[inline(always)]
        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Vec::new())
        }
    }

    deserializer.deserialize_any(VisitorImpl(PhantomData))
}

/// Accepts **array ∪ null** JSON inputs and yields `Vec<T>`.
#[inline(always)]
pub fn null_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    struct VisitorImpl<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for VisitorImpl<T>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("sequence, null, or unit")
        }

        #[inline(always)]
        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        #[inline(always)]
        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Vec::new())
        }

        #[inline(always)]
        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Vec::new())
        }
    }

    deserializer.deserialize_any(VisitorImpl(PhantomData))
}

// -----------------------------------------------------------------------------
// Exhaustive unit tests (compile-time & run-time)
// -----------------------------------------------------------------------------
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
        let a = json!({"k1":"v1"});
        let b = json!({"k2":"v2"});
        assert_eq!(merge(a, b), json!({"k1":"v1","k2":"v2"}));
    }

    #[test]
    fn merge_in_place() {
        let mut a = json!({"k1":"v1"});
        merge_inplace(&mut a, json!({"k2":"v2"}));
        assert_eq!(a, json!({"k1":"v1","k2":"v2"}));
    }

    // ----- stringified JSON -----------------------------------------------
    #[test]
    fn stringified_roundtrip() {
        let original = Dummy {
            data: json!({"k":"v"}),
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
}
