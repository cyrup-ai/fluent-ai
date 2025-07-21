//! Canonical macros for the streams-only architecture.

/// Emits a value into the stream.
///
/// This macro is a lightweight wrapper around the sender's `send` method,
/// providing a consistent and ergonomic way to produce values in a stream.
/// It gracefully handles the case where the receiver has been dropped.
#[macro_export]
macro_rules! emit {
    ($sender:expr, $value:expr) => {
        if $sender.send($value).is_err() {
            // The receiver has been dropped, so we can gracefully stop.
            // This is not an error condition, but a natural end to the stream.
            return;
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
        log::error!("Stream error in {}: {}. Details: {}", file!(), $context, $err);
        return;
    }};
}
