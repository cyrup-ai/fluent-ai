//! Domain-polished builder DSL for collection construction & transformation

// Re-exported prelude to keep userland simple
pub mod prelude {
    pub use super::{Builder, MapExt, SetExt};
    pub use hashbrown::{HashMap, HashSet};
    pub use indexmap::IndexMap;
    pub use std::collections::{BTreeMap, BTreeSet};
}

use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;

//──────────────────────────────────────────────────────────────────────────────
// Builder API — only accepts closures (macro hidden internally)
//──────────────────────────────────────────────────────────────────────────────

pub struct Builder;

impl Builder {
    pub fn map<K: Eq + Hash, V>(f: impl FnOnce() -> HashMap<K, V>) -> HashMap<K, V> {
        f()
    }

    pub fn map_indexed<K: Eq + Hash, V>(f: impl FnOnce() -> IndexMap<K, V>) -> IndexMap<K, V> {
        f()
    }

    pub fn map_ordered<K: Ord, V>(f: impl FnOnce() -> BTreeMap<K, V>) -> BTreeMap<K, V> {
        f()
    }

    pub fn set<T: Eq + Hash>(f: impl FnOnce() -> HashSet<T>) -> HashSet<T> {
        f()
    }

    pub fn set_ordered<T: Ord>(f: impl FnOnce() -> BTreeSet<T>) -> BTreeSet<T> {
        f()
    }
}

//──────────────────────────────────────────────────────────────────────────────
// Map Extensions — composable transformation methods for all map types
//──────────────────────────────────────────────────────────────────────────────

pub trait MapExt<K, V>: Sized {
    fn map_values<U>(self, f: impl FnMut(V) -> U) -> HashMap<K, U>;
    fn map_keys<U: Eq + Hash>(self, f: impl FnMut(K) -> U) -> HashMap<U, V>;
    fn filter(self, f: impl FnMut(&(K, V)) -> bool) -> HashMap<K, V>;
    fn reduce<U>(self, init: U, f: impl FnMut(U, (K, V)) -> U) -> U;
    fn to_ordered(self) -> BTreeMap<K, V>
    where
        K: Ord;
    fn to_indexed(self) -> IndexMap<K, V>
    where
        K: Eq + Hash;
    fn slice(self, range: std::ops::Range<usize>) -> HashMap<K, V>;
    fn take(self, n: usize) -> HashMap<K, V>;
    fn pop_first(self) -> Option<(K, V)>;
    fn pop_last(self) -> Option<(K, V)>;
    fn into_indexed_by<F, Q: Eq + Hash>(self, f: F) -> IndexMap<Q, V>
    where
        F: Fn(&K, &V) -> Q;
    fn partition(self, chunk_size: usize) -> Vec<HashMap<K, V>>;

    fn map_ok<U, E>(self, f: impl FnMut(V) -> Result<U, E>) -> Result<HashMap<K, U>, E>;
    fn map_err<E, F: FnMut((K, V)) -> Result<(K, V), E>>(self, f: F) -> Result<HashMap<K, V>, E>;
    fn validate_each<E>(self, f: impl FnMut(&K, &V) -> Result<(), E>) -> Result<Self, E>;

    fn tap_each(self, f: impl FnMut(&K, &V)) -> Self;
    fn tee_each(self, f: impl FnMut(K, V)) -> Self
    where
        K: Clone,
        V: Clone;
}

impl<K: Eq + Hash + Clone, V: Clone> MapExt<K, V> for HashMap<K, V> {
    fn map_values<U>(self, mut f: impl FnMut(V) -> U) -> HashMap<K, U> {
        self.into_iter().map(|(k, v)| (k, f(v))).collect()
    }

    fn map_keys<U: Eq + Hash>(self, mut f: impl FnMut(K) -> U) -> HashMap<U, V> {
        self.into_iter().map(|(k, v)| (f(k), v)).collect()
    }

    fn filter(self, mut f: impl FnMut(&(K, V)) -> bool) -> HashMap<K, V> {
        self.into_iter().filter(|kv| f(kv)).collect()
    }

    fn reduce<U>(self, init: U, f: impl FnMut(U, (K, V)) -> U) -> U {
        self.into_iter().fold(init, f)
    }

    fn to_ordered(self) -> BTreeMap<K, V>
    where
        K: Ord,
    {
        self.into_iter().collect()
    }

    fn to_indexed(self) -> IndexMap<K, V>
    where
        K: Eq + Hash,
    {
        self.into_iter().collect()
    }

    fn slice(self, range: std::ops::Range<usize>) -> HashMap<K, V> {
        self.into_iter()
            .skip(range.start)
            .take(range.end - range.start)
            .collect()
    }

    fn take(self, n: usize) -> HashMap<K, V> {
        self.into_iter().take(n).collect()
    }

    fn pop_first(mut self) -> Option<(K, V)> {
        self.drain().next()
    }

    fn pop_last(mut self) -> Option<(K, V)> {
        let mut last = None;
        for item in self.drain() {
            last = Some(item);
        }
        last
    }

    fn into_indexed_by<F, Q: Eq + Hash>(self, f: F) -> IndexMap<Q, V>
    where
        F: Fn(&K, &V) -> Q,
    {
        self.into_iter().map(|(k, v)| (f(&k, &v), v)).collect()
    }

    fn partition(self, chunk_size: usize) -> Vec<HashMap<K, V>> {
        let mut acc = Vec::new();
        for (i, (k, v)) in self.into_iter().enumerate() {
            if i % chunk_size == 0 {
                acc.push(HashMap::new());
            }
            if let Some(last) = acc.last_mut() {
                last.insert(k, v);
            }
        }
        acc
    }

    fn map_ok<U, E>(self, mut f: impl FnMut(V) -> Result<U, E>) -> Result<HashMap<K, U>, E> {
        self.into_iter()
            .map(|(k, v)| f(v).map(|u| (k, u)))
            .collect()
    }

    fn map_err<E, F: FnMut((K, V)) -> Result<(K, V), E>>(self, f: F) -> Result<HashMap<K, V>, E> {
        self.into_iter().map(f).collect()
    }

    fn validate_each<E>(self, mut f: impl FnMut(&K, &V) -> Result<(), E>) -> Result<Self, E> {
        for (k, v) in &self {
            f(k, v)?;
        }
        Ok(self)
    }

    fn tap_each(self, mut f: impl FnMut(&K, &V)) -> Self {
        for (k, v) in &self {
            f(k, v);
        }
        self
    }

    fn tee_each(self, mut f: impl FnMut(K, V)) -> Self
    where
        K: Clone,
        V: Clone,
    {
        self.into_iter()
            .inspect(|(k, v)| f(k.clone(), v.clone()))
            .collect()
    }
}

// MapExt implementation for BTreeMap
impl<K: Ord + Clone + Hash, V: Clone> MapExt<K, V> for BTreeMap<K, V> {
    fn map_values<U>(self, mut f: impl FnMut(V) -> U) -> HashMap<K, U> {
        self.into_iter().map(|(k, v)| (k, f(v))).collect()
    }

    fn map_keys<U: Eq + Hash>(self, mut f: impl FnMut(K) -> U) -> HashMap<U, V> {
        self.into_iter().map(|(k, v)| (f(k), v)).collect()
    }

    fn filter(self, mut f: impl FnMut(&(K, V)) -> bool) -> HashMap<K, V> {
        self.into_iter().filter(|kv| f(kv)).collect()
    }

    fn reduce<U>(self, init: U, f: impl FnMut(U, (K, V)) -> U) -> U {
        self.into_iter().fold(init, f)
    }

    fn to_ordered(self) -> BTreeMap<K, V>
    where
        K: Ord,
    {
        self
    }

    fn to_indexed(self) -> IndexMap<K, V>
    where
        K: Eq + Hash,
    {
        self.into_iter().collect()
    }

    fn slice(self, range: std::ops::Range<usize>) -> HashMap<K, V> {
        self.into_iter()
            .skip(range.start)
            .take(range.end - range.start)
            .collect()
    }

    fn take(self, n: usize) -> HashMap<K, V> {
        self.into_iter().take(n).collect()
    }

    fn pop_first(self) -> Option<(K, V)> {
        self.into_iter().next()
    }

    fn pop_last(self) -> Option<(K, V)> {
        self.into_iter().next_back()
    }

    fn into_indexed_by<F, Q: Eq + Hash>(self, f: F) -> IndexMap<Q, V>
    where
        F: Fn(&K, &V) -> Q,
    {
        self.into_iter().map(|(k, v)| (f(&k, &v), v)).collect()
    }

    fn partition(self, chunk_size: usize) -> Vec<HashMap<K, V>> {
        let mut acc = Vec::new();
        for (i, (k, v)) in self.into_iter().enumerate() {
            if i % chunk_size == 0 {
                acc.push(HashMap::new());
            }
            if let Some(last) = acc.last_mut() {
                last.insert(k, v);
            }
        }
        acc
    }

    fn map_ok<U, E>(self, mut f: impl FnMut(V) -> Result<U, E>) -> Result<HashMap<K, U>, E> {
        self.into_iter()
            .map(|(k, v)| f(v).map(|u| (k, u)))
            .collect()
    }

    fn map_err<E, F: FnMut((K, V)) -> Result<(K, V), E>>(self, f: F) -> Result<HashMap<K, V>, E> {
        self.into_iter().map(f).collect()
    }

    fn validate_each<E>(self, mut f: impl FnMut(&K, &V) -> Result<(), E>) -> Result<Self, E> {
        for (k, v) in &self {
            f(k, v)?;
        }
        Ok(self)
    }

    fn tap_each(self, mut f: impl FnMut(&K, &V)) -> Self {
        for (k, v) in &self {
            f(k, v);
        }
        self
    }

    fn tee_each(self, mut f: impl FnMut(K, V)) -> Self
    where
        K: Clone,
        V: Clone,
    {
        self.into_iter()
            .inspect(|(k, v)| f(k.clone(), v.clone()))
            .collect()
    }
}

// MapExt implementation for IndexMap
impl<K: Eq + Hash + Clone, V: Clone> MapExt<K, V> for IndexMap<K, V> {
    fn map_values<U>(self, mut f: impl FnMut(V) -> U) -> HashMap<K, U> {
        self.into_iter().map(|(k, v)| (k, f(v))).collect()
    }

    fn map_keys<U: Eq + Hash>(self, mut f: impl FnMut(K) -> U) -> HashMap<U, V> {
        self.into_iter().map(|(k, v)| (f(k), v)).collect()
    }

    fn filter(self, mut f: impl FnMut(&(K, V)) -> bool) -> HashMap<K, V> {
        self.into_iter().filter(|kv| f(kv)).collect()
    }

    fn reduce<U>(self, init: U, f: impl FnMut(U, (K, V)) -> U) -> U {
        self.into_iter().fold(init, f)
    }

    fn to_ordered(self) -> BTreeMap<K, V>
    where
        K: Ord,
    {
        self.into_iter().collect()
    }

    fn to_indexed(self) -> IndexMap<K, V>
    where
        K: Eq + Hash,
    {
        self
    }

    fn slice(self, range: std::ops::Range<usize>) -> HashMap<K, V> {
        self.into_iter()
            .skip(range.start)
            .take(range.end - range.start)
            .collect()
    }

    fn take(self, n: usize) -> HashMap<K, V> {
        self.into_iter().take(n).collect()
    }

    fn pop_first(mut self) -> Option<(K, V)> {
        self.shift_remove_index(0)
    }

    fn pop_last(mut self) -> Option<(K, V)> {
        let len = self.len();
        if len > 0 {
            self.shift_remove_index(len - 1)
        } else {
            None
        }
    }

    fn into_indexed_by<F, Q: Eq + Hash>(self, f: F) -> IndexMap<Q, V>
    where
        F: Fn(&K, &V) -> Q,
    {
        self.into_iter().map(|(k, v)| (f(&k, &v), v)).collect()
    }

    fn partition(self, chunk_size: usize) -> Vec<HashMap<K, V>> {
        let mut acc = Vec::new();
        for (i, (k, v)) in self.into_iter().enumerate() {
            if i % chunk_size == 0 {
                acc.push(HashMap::new());
            }
            if let Some(last) = acc.last_mut() {
                last.insert(k, v);
            }
        }
        acc
    }

    fn map_ok<U, E>(self, mut f: impl FnMut(V) -> Result<U, E>) -> Result<HashMap<K, U>, E> {
        self.into_iter()
            .map(|(k, v)| f(v).map(|u| (k, u)))
            .collect()
    }

    fn map_err<E, F: FnMut((K, V)) -> Result<(K, V), E>>(self, f: F) -> Result<HashMap<K, V>, E> {
        self.into_iter().map(f).collect()
    }

    fn validate_each<E>(self, mut f: impl FnMut(&K, &V) -> Result<(), E>) -> Result<Self, E> {
        for (k, v) in &self {
            f(k, v)?;
        }
        Ok(self)
    }

    fn tap_each(self, mut f: impl FnMut(&K, &V)) -> Self {
        for (k, v) in &self {
            f(k, v);
        }
        self
    }

    fn tee_each(self, mut f: impl FnMut(K, V)) -> Self
    where
        K: Clone,
        V: Clone,
    {
        self.into_iter()
            .inspect(|(k, v)| f(k.clone(), v.clone()))
            .collect()
    }
}

//──────────────────────────────────────────────────────────────────────────────
// Set Extensions — composable transformation methods for all set types
//──────────────────────────────────────────────────────────────────────────────

pub trait SetExt<T>: Sized {
    fn map<U: Eq + Hash>(self, f: impl FnMut(T) -> U) -> HashSet<U>;
    fn filter(self, f: impl FnMut(&T) -> bool) -> HashSet<T>
    where
        T: Eq + Hash;
    fn to_vec(self) -> Vec<T>;
    fn to_ordered(self) -> BTreeSet<T>
    where
        T: Ord;
    fn slice(self, range: std::ops::Range<usize>) -> HashSet<T>
    where
        T: Eq + Hash;
    fn take(self, n: usize) -> HashSet<T>
    where
        T: Eq + Hash;
    fn pop_first(self) -> Option<T>;
    fn pop_last(self) -> Option<T>;
    fn partition(self, chunk_size: usize) -> Vec<HashSet<T>>
    where
        T: Eq + Hash;

    fn map_ok<U, E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<HashSet<U>, E>
    where
        U: Eq + Hash;
    fn validate_each<E>(self, f: impl FnMut(&T) -> Result<(), E>) -> Result<Self, E>;
    fn tap_each(self, f: impl FnMut(&T)) -> Self;
    fn tee_each(self, f: impl FnMut(T)) -> Self
    where
        T: Clone;
}

impl<T: Eq + Hash + Clone> SetExt<T> for HashSet<T> {
    fn map<U: Eq + Hash>(self, f: impl FnMut(T) -> U) -> HashSet<U> {
        self.into_iter().map(f).collect()
    }

    fn filter(self, mut f: impl FnMut(&T) -> bool) -> HashSet<T> {
        self.into_iter().filter(|t| f(t)).collect()
    }

    fn to_vec(self) -> Vec<T> {
        self.into_iter().collect()
    }

    fn to_ordered(self) -> BTreeSet<T>
    where
        T: Ord,
    {
        self.into_iter().collect()
    }

    fn slice(self, range: std::ops::Range<usize>) -> HashSet<T> {
        self.into_iter()
            .skip(range.start)
            .take(range.end - range.start)
            .collect()
    }

    fn take(self, n: usize) -> HashSet<T> {
        self.into_iter().take(n).collect()
    }

    fn pop_first(mut self) -> Option<T> {
        self.drain().next()
    }

    fn pop_last(mut self) -> Option<T> {
        let mut last = None;
        for item in self.drain() {
            last = Some(item);
        }
        last
    }

    fn partition(self, chunk_size: usize) -> Vec<HashSet<T>> {
        let mut acc = Vec::new();
        for (i, item) in self.into_iter().enumerate() {
            if i % chunk_size == 0 {
                acc.push(HashSet::new());
            }
            if let Some(last) = acc.last_mut() {
                last.insert(item);
            }
        }
        acc
    }

    fn map_ok<U, E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<HashSet<U>, E>
    where
        U: Eq + Hash,
    {
        self.into_iter().map(f).collect()
    }

    fn validate_each<E>(self, mut f: impl FnMut(&T) -> Result<(), E>) -> Result<Self, E> {
        for t in &self {
            f(t)?;
        }
        Ok(self)
    }

    fn tap_each(mut self, mut f: impl FnMut(&T)) -> Self {
        for t in &self {
            f(t);
        }
        self
    }

    fn tee_each(self, mut f: impl FnMut(T)) -> Self {
        self.into_iter().inspect(|t| f(t.clone())).collect()
    }
}
