// ============================================================================
// File: src/workflow/conditional.rs
// ----------------------------------------------------------------------------
// Enum-dispatch helpers for branching pipelines.
//
// • Generates zero-cost structs at compile-time (no boxing, no heap).
// • Macros now point at `workflow` instead of the old `pipeline` paths.
// ============================================================================

// Internal macro - not exported
macro_rules! conditional {
    ($enum:ident, $( $variant:ident => $op:expr ),+ $(,)?) => {{
        #[allow(non_camel_case_types)]
        struct ConditionalOp<$($variant),+> {
            $(
                #[allow(non_snake_case)]
                $variant: $variant,
            )+
        }

        impl<Value, Out, $($variant),+> $crate::workflow::Op for ConditionalOp<$($variant),+>
        where
            $( $variant: $crate::workflow::Op<Input = Value, Output = Out> ),+,
            Value: Send + Sync,
            Out:   Send + Sync,
        {
            type Input  = $enum<Value>;
            type Output = Out;

            async fn call(&self, input: Self::Input) -> Self::Output {
                match input {
                    $(
                        $enum::$variant(v) => self.$variant.call(v).await,
                    )+
                }
            }
        }

        ConditionalOp { $($variant: $op),+ }
    }};
}

// Internal macro - not exported
macro_rules! try_conditional {
    ($enum:ident, $( $variant:ident => $op:expr ),+ $(,)?) => {{
        #[allow(non_camel_case_types)]
        struct TryConditionalOp<$($variant),+> {
            $(
                #[allow(non_snake_case)]
                $variant: $variant,
            )+
        }

        impl<Value, Out, Err, $($variant),+> $crate::workflow::TryOp
            for TryConditionalOp<$($variant),+>
        where
            $( $variant: $crate::workflow::TryOp<Input = Value, Output = Out, Error = Err> ),+,
            Value: Send + Sync,
            Out:   Send + Sync,
            Err:   Send + Sync,
        {
            type Input  = $enum<Value>;
            type Output = Out;
            type Error  = Err;

            async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
                match input {
                    $(
                        $enum::$variant(v) => self.$variant.try_call(v).await,
                    )+
                }
            }
        }

        TryConditionalOp { $($variant: $op),+ }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::workflow::map;

    #[tokio::test]
    async fn conditional_dispatches_correctly() {
        enum E<T> {
            A(T),
            B(T),
        }

        let add = map(|x: i32| x + 1);
        let mul = map(|x: i32| x * 2);

        let op = conditional!(E,
            A => add,
            B => mul,
        );

        assert_eq!(op.call(E::A(5)).await, 6);
        assert_eq!(op.call(E::B(5)).await, 10);
    }

    #[tokio::test]
    async fn try_conditional_propagates_errors() {
        enum E<T> {
            A(T),
            B(T),
        }

        let ok = map(|x: i32| Ok::<_, &str>(x + 1));
        let err = map(|_| Err::<i32, &str>("nope"));

        let op = try_conditional!(E,
            A => ok,
            B => err,
        );

        assert_eq!(op.try_call(E::A(1)).await, Ok(2));
        assert_eq!(op.try_call(E::B(0)).await, Err("nope"));
    }
}
