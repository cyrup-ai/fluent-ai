// ============================================================================
// File: src/workflow/parallel.rs
// ----------------------------------------------------------------------------
// Fully-static, allocation-free fan-out combinator + macro helpers.
//
//   • `Parallel`      – runs two independent Ops concurrently.
//   • `parallel!`     – variadic macro building an N-way Parallel tree.
//   • `try_parallel!` – same but propagates the first error.
//
// All macros flatten the nested output tuple so users get a clean `(A,B,C,…)`.
// Every public function is `#[inline]` to let LLVM fuse trivial layers.
// ============================================================================

use futures::{join, try_join};

use super::{Op, TryOp};

// ================================================================
// 0. Two-way primitive used by the variadic macros
// ================================================================
pub struct Parallel<A, B> {
    left: A,
    right: B,
}

impl<A, B> Parallel<A, B> {
    #[inline(always)]
    pub fn new(left: A, right: B) -> Self {
        Self { left, right }
    }
}

/* --------------------------------------------------------------------------
 * Op implementation – returns both outputs as a tuple.
 * ----------------------------------------------------------------------- */
impl<A, B> Op for Parallel<A, B>
where
    A: Op,
    A::Input: Clone,
    B: Op<Input = A::Input>,
{
    type Input = A::Input;
    type Output = (A::Output, B::Output);

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        join!(self.left.call(input.clone()), self.right.call(input))
    }
}

/* --------------------------------------------------------------------------
 * TryOp implementation – short-circuits on the first error.
 * ----------------------------------------------------------------------- */
impl<A, B> TryOp for Parallel<A, B>
where
    A: TryOp,
    A::Input: Clone,
    B: TryOp<Input = A::Input, Error = A::Error>,
{
    type Input = A::Input;
    type Output = (A::Output, B::Output);
    type Error = A::Error;

    #[inline]
    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        try_join!(
            self.left.try_call(input.clone()),
            self.right.try_call(input)
        )
    }
}

// ================================================================
// 1. Variadic parallel! and try_parallel! macros
// ================================================================
// Internal macro - not exported
macro_rules! parallel_op {
    ($a:expr, $b:expr) => {
        $crate::workflow::parallel::Parallel::new($a, $b)
    };
    ($a:expr, $($rest:expr),+ $(,)?) => {
        $crate::workflow::parallel::Parallel::new($a, $crate::parallel_op!($($rest),+))
    };
}

// Internal macro - not exported
macro_rules! tuple_pick {
    ($id:ident +) => { $id };
    ($id:ident)   => { ($id, ..) };
    ($id:ident _ $($rest:tt)*) => { (_, $crate::tuple_pick!($id $($rest)*)) };
}

// Internal macro - not exported
macro_rules! parallel_internal {
    (
        stack:  [ $($pos:tt)* ]
        accum:  [ $( $expr:expr => ( $($idx:tt)* ) ),* ]
        pending:[]
    ) => {
        $crate::parallel_op!( $( $expr ),* ).map(|out| {
            ( $( {
                let $crate::tuple_pick!(v $($idx)*) = out;
                v
            } ),* )
        })
    };
    (
        stack:  [ $($pos:tt)* ]
        accum:  [ $( $expr:expr => ( $($idx:tt)* ) ),* ]
        pending:[ $head:expr $( , $tail:expr )* ]
    ) => {
        $crate::parallel_internal!(
            stack: [ $($pos)* _ ]
            accum: [ $( $expr => ( $($idx)* ) , )* $head => ( $($pos)* ) ]
            pending:[ $( $tail ),* ]
        )
    };
}

// Internal macro - not exported
macro_rules! parallel {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        $crate::parallel_internal!(
            stack:   []
            accum:   []
            pending: [ $first $(, $rest)* ]
        )
    };
}

/* ------------- try_parallel! (error-propagating variant) ----------------- */
// Internal macro - not exported
macro_rules! try_parallel_internal {
    (
        stack:  [ $($pos:tt)* ]
        accum:  [ $( $expr:expr => ( $($idx:tt)* ) ),* ]
        pending:[]
    ) => {
        $crate::parallel_op!( $( $expr ),* ).map_ok(|out| {
            ( $( {
                let $crate::tuple_pick!(v $($idx)*) = out;
                v
            } ),* )
        })
    };
    (
        stack:  [ $($pos:tt)* ]
        accum:  [ $( $expr:expr => ( $($idx:tt)* ) ),* ]
        pending:[ $head:expr $( , $tail:expr )* ]
    ) => {
        $crate::try_parallel_internal!(
            stack: [ $($pos)* _ ]
            accum: [ $( $expr => ( $($idx)* ) , )* $head => ( $($pos)* ) ]
            pending:[ $( $tail ),* ]
        )
    };
}

// Internal macro - not exported
macro_rules! try_parallel {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        $crate::try_parallel_internal!(
            stack:   []
            accum:   []
            pending: [ $first $(, $rest)* ]
        )
    };
}

// Re-export macros for downstream users.
pub use {parallel, try_parallel};

// ================================================================
// 2. Tests
// ================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::workflow::{map, passthrough, then};
    use crate::workflow::{self, Op, TryOp};

    #[tokio::test]
    async fn two_way_parallel() {
        let out = Parallel::new(map(|x| x + 1), map(|x| x * 3)).call(2).await;
        assert_eq!(out, (3, 6));
    }

    #[tokio::test]
    async fn macro_parallel_variadic() {
        let op = parallel!(
            passthrough(),
            map(|x: i32| x + 1),
            map(|x: i32| format!("{x}!"))
        );
        assert_eq!(op.call(1).await, (1, 2, "1!".into()));
    }

    #[tokio::test]
    async fn try_parallel_ok() {
        let op = try_parallel!(
            map(|x: i32| Ok::<_, &str>(x + 1)),
            map(|x: i32| Ok::<_, &str>(x * 2))
        );
        assert_eq!(op.try_call(3).await, Ok((4, 6)));
    }

    #[tokio::test]
    async fn try_parallel_err() {
        let op = try_parallel!(
            map(|_: i32| Err::<i32, _>("boom")),
            map(|x: i32| Ok::<_, &str>(x * 2))
        );
        assert_eq!(op.try_call(3).await, Err("boom"));
    }

    #[tokio::test]
    async fn parallel_in_chain() {
        let chain = passthrough()
            .then(|x| async move { parallel!(passthrough(), map(|y: i32| y * 2)).call(x).await })
            .map(|(a, b)| a + b);

        assert_eq!(chain.call(4).await, 12);
    }
}
