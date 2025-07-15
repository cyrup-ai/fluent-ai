// ============================================================================
// File: src/one_or_many.rs
// ----------------------------------------------------------------------------
// Compact container that guarantees ≥ 1 element.
//
// • Zero-alloc fast-path; no internal heap indirections beyond `Vec<T>`.
// • Iterator trio (ref / mut / owned) is allocation-free and branch-predictable.
// • Custom serde so JSON      → OneOrMany maps intuitively:
//
//     "val"      => ["val"]
//     ["a","b"]  => ["a","b"]
//     null       => Option::None                 (via helper)
// ============================================================================

#![allow(clippy::len_without_is_empty)]

use serde::{
    de::{self, Deserializer, MapAccess, SeqAccess, Visitor},
    ser::{SerializeSeq, Serializer},
    Deserialize, Serialize,
};
use std::{convert::Infallible, fmt, marker::PhantomData, str::FromStr};

// -----------------------------------------------------------------------------
// Core type
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneOrMany<T> {
    first: T,
    rest: Vec<T>,
}

// -----------------------------------------------------------------------------
// Public API – zero-alloc fast-path
// -----------------------------------------------------------------------------
impl<T: Clone> OneOrMany<T> {
    #[inline(always)]
    pub fn one(item: T) -> Self {
        Self {
            first: item,
            rest: Vec::new(),
        }
    }

    pub fn many<I>(items: I) -> Result<Self, EmptyListError>
    where
        I: IntoIterator<Item = T>,
    {
        let mut it = items.into_iter();
        let first = it.next().ok_or(EmptyListError)?;
        Ok(Self {
            first,
            rest: it.collect(),
        })
    }

    #[inline(always)]
    pub fn first(&self) -> T {
        self.first.clone()
    }
    #[inline(always)]
    pub fn rest(&self) -> Vec<T> {
        self.rest.clone()
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        1 + self.rest.len()
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        false
    }

    #[inline(always)]
    pub fn push(&mut self, item: T) {
        self.rest.push(item)
    }

    pub fn insert(&mut self, index: usize, item: T) {
        if index == 0 {
            let old = std::mem::replace(&mut self.first, item);
            self.rest.insert(0, old);
        } else {
            self.rest.insert(index - 1, item);
        }
    }

    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> OneOrMany<U> {
        OneOrMany {
            first: f(self.first),
            rest: self.rest.into_iter().map(f).collect(),
        }
    }

    pub fn try_map<U, E, F: FnMut(T) -> Result<U, E>>(self, mut f: F) -> Result<OneOrMany<U>, E> {
        Ok(OneOrMany {
            first: f(self.first)?,
            rest: self.rest.into_iter().map(f).collect::<Result<_, _>>()?,
        })
    }

    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            first: Some(&self.first),
            rest: self.rest.iter(),
        }
    }
    #[inline(always)]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            first: Some(&mut self.first),
            rest: self.rest.iter_mut(),
        }
    }

    pub fn merge<I>(src: I) -> Result<Self, EmptyListError>
    where
        I: IntoIterator<Item = OneOrMany<T>>,
    {
        let mut it = src.into_iter().flat_map(OneOrMany::into_iter);
        let first = it.next().ok_or(EmptyListError)?;
        Ok(Self {
            first,
            rest: it.collect(),
        })
    }
}

// -----------------------------------------------------------------------------
// Error
// -----------------------------------------------------------------------------
#[derive(Debug, thiserror::Error)]
#[error("Cannot create OneOrMany with an empty list")]
pub struct EmptyListError;

// -----------------------------------------------------------------------------
// Iterators
// -----------------------------------------------------------------------------
pub struct Iter<'a, T> {
    first: Option<&'a T>,
    rest: std::slice::Iter<'a, T>,
}
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.first.take().or_else(|| self.rest.next())
    }
}

pub struct IterMut<'a, T> {
    first: Option<&'a mut T>,
    rest: std::slice::IterMut<'a, T>,
}
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(r) = self.first.take() {
            Some(r)
        } else {
            self.rest.next()
        }
    }
}

pub struct IntoIter<T> {
    first: Option<T>,
    rest: std::vec::IntoIter<T>,
}
impl<T> Iterator for IntoIter<T> {
    type Item = T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.first.take().or_else(|| self.rest.next())
    }
}
impl<T> IntoIterator for OneOrMany<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            first: Some(self.first),
            rest: self.rest.into_iter(),
        }
    }
}

// -----------------------------------------------------------------------------
// Serde impls – array only (always ≥1 element)
// -----------------------------------------------------------------------------
impl<T> Serialize for OneOrMany<T>
where
    T: Serialize + Clone,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        seq.serialize_element(&self.first)?;
        for item in &self.rest {
            seq.serialize_element(item)?;
        }
        seq.end()
    }
}

impl<'de, T> Deserialize<'de> for OneOrMany<T>
where
    T: Deserialize<'de> + Clone,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V<T>(PhantomData<T>);
        impl<'de, T> Visitor<'de> for V<T>
        where
            T: Deserialize<'de> + Clone,
        {
            type Value = OneOrMany<T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a non-empty sequence or single value")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let first = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let mut rest = Vec::new();
                while let Some(v) = seq.next_element()? {
                    rest.push(v);
                }
                Ok(OneOrMany { first, rest })
            }
            fn visit_map<M>(self, map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                Ok(OneOrMany::one(Deserialize::deserialize(
                    de::value::MapAccessDeserializer::new(map),
                )?))
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(OneOrMany::one(v.parse().map_err(E::custom)?))
            }
        }
        deserializer.deserialize_any(V(PhantomData))
    }
}

// -----------------------------------------------------------------------------
// Helper deserialisers for struct fields
// -----------------------------------------------------------------------------

/// Accepts `"val"` **or** `["val", …]` → `OneOrMany`.
pub fn string_or_one_or_many<'de, T, D>(d: D) -> Result<OneOrMany<T>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
    D: Deserializer<'de>,
{
    struct V<T>(PhantomData<T>);
    impl<'de, T> Visitor<'de> for V<T>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
    {
        type Value = OneOrMany<T>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("string or sequence")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(OneOrMany::one(v.parse().map_err(E::custom)?))
        }
        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
        fn visit_map<M>(self, map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            Ok(OneOrMany::one(Deserialize::deserialize(
                de::value::MapAccessDeserializer::new(map),
            )?))
        }
    }
    d.deserialize_any(V(PhantomData))
}

/// Accepts `null`, `"val"`, or `["val", …]` → `Option<OneOrMany<T>>`.
pub fn string_or_option_one_or_many<'de, T, D>(d: D) -> Result<Option<OneOrMany<T>>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
    D: Deserializer<'de>,
{
    struct V<T>(PhantomData<T>);
    impl<'de, T> Visitor<'de> for V<T>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
    {
        type Value = Option<OneOrMany<T>>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("null, string or sequence")
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }
        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }
        fn visit_some<D>(self, d: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            string_or_one_or_many(d).map(Some)
        }
    }
    d.deserialize_option(V(PhantomData))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn merge_and_len() {
        let merged =
            OneOrMany::merge([OneOrMany::one(1), OneOrMany::many(vec![2, 3]).unwrap()]).unwrap();
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn serde_roundtrip_single() {
        let original = OneOrMany::one("x".to_string());
        let s = serde_json::to_string(&original).unwrap();
        assert_eq!(s, r#"["x"]"#);
        let parsed: OneOrMany<String> = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn field_helper_variants() {
        #[derive(Deserialize)]
        struct W {
            #[serde(deserialize_with = "string_or_option_one_or_many")]
            v: Option<OneOrMany<i32>>,
        }
        assert!(serde_json::from_str::<W>(r#"{ "v": null }"#)
            .unwrap()
            .v
            .is_none());
        assert_eq!(
            serde_json::from_str::<W>(r#"{ "v": "3" }"#)
                .unwrap()
                .v
                .unwrap()
                .first(),
            3
        );
    }

    #[test]
    fn from_str_map() {
        let raw = json!({"k":1});
        let one: OneOrMany<serde_json::Value> = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(one.first(), raw);
    }
}
