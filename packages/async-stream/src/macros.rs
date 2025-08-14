//! Canonical macros for the streams-only architecture.

/// Emits a value into the stream.
///
/// This macro is a lightweight wrapper around the sender's `send` method,
/// providing a consistent and ergonomic way to produce values in a stream.
/// It gracefully handles the case where the receiver has been dropped.
#[macro_export]
macro_rules! emit {
    ($sender:expr, $value:expr) => {
        if let Err(_) = $sender.send($value) {
            // The receiver has been dropped, so we can gracefully stop.
            // This is not an error condition, but a natural end to the stream.
            return;
        }
    };
}

/// Pattern matching syntax for on_chunk processing with arbitrary user code
///
/// Transforms `|chunk| { Ok => { arbitrary_code; expr }, Err(e) => { more_code; expr } }`
/// into proper match syntax. Supports multi-statement blocks and complex expressions.
#[macro_export]
macro_rules! on_chunk_pattern {
    // Single expression version
    (|$param:ident| { Ok => $ok_expr:expr, Err($err:ident) => $err_expr:expr $(,)? }) => {
        |$param| match $param {
            Ok(val) => {
                let $param = val;
                $ok_expr
            }
            Err($err) => $err_expr,
        }
    };

    // Block expression version for arbitrary code
    (|$param:ident| { Ok => $ok_block:block, Err($err:ident) => $err_block:block $(,)? }) => {
        |$param| match $param {
            Ok(val) => {
                let $param = val;
                $ok_block
            }
            Err($err) => $err_block,
        }
    };
}

/// Handles an error within a stream-producing task.
///
/// This macro logs the error with a consistent format and then gracefully
/// terminates the task. It ensures that all errors are reported without
/// causing a panic.
#[macro_export]
macro_rules! handle_error {
    ($err:expr, $context:expr) => {{
        log::error!(
            "Stream error in {}: {}. Details: {}",
            file!(),
            $context,
            $err
        );
        return;
    }};
}
