//! WASM-specific HTTP client implementation
//!
//! This module provides WebAssembly-compatible HTTP functionality using the Fetch API.
//! It's designed to work in browser environments and web workers.

use std::convert::TryInto;
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use js_sys::Function;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::{Closure, wasm_bindgen};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
// Removed JsFuture and spawn_local - using AsyncStream only
#[cfg(target_arch = "wasm32")]
use web_sys::{AbortController, AbortSignal};

// WASM utility functions
#[cfg(target_arch = "wasm32")]
fn js_fetch(request: web_sys::Request) -> js_sys::Promise {
    web_sys::window().unwrap().fetch_with_request(&request)
}

#[cfg(target_arch = "wasm32")]
fn set_timeout(callback: &js_sys::Function, delay: i32) -> i32 {
    web_sys::window().unwrap().set_timeout_with_callback_and_timeout_and_arguments_0(callback, delay).unwrap()
}

#[cfg(target_arch = "wasm32")]
fn clear_timeout(id: i32) {
    web_sys::window().unwrap().clear_timeout_with_handle(id);
}

// Re-export WASM types from web-sys
#[cfg(target_arch = "wasm32")]
pub use web_sys::{AbortController, AbortSignal};

// For non-WASM targets, provide stub types
#[cfg(not(target_arch = "wasm32"))]
pub struct AbortController;
#[cfg(not(target_arch = "wasm32"))]
pub struct AbortSignal;

#[cfg(target_arch = "wasm32")]
use std::fmt;

#[cfg(target_arch = "wasm32")]
use bytes::Bytes;
#[cfg(target_arch = "wasm32")]
use http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
#[cfg(target_arch = "wasm32")]
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use url::Url;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
// Removed JsFuture - using AsyncStream only
use js_sys::Promise;
use fluent_ai_async::prelude::MessageChunk;

use crate::error::Error;

// MessageChunk implementation for JsValue
impl MessageChunk for JsValue {
    fn bad_chunk(error: String) -> Self {
        JsValue::from_str(&format!("ERROR: {}", error))
    }

    fn error(&self) -> Option<&str> {
        if let Some(s) = self.as_string() {
            if s.starts_with("ERROR: ") {
                return Some(&s[7..]);
            }
        }
        None
    }

    fn is_error(&self) -> bool {
        if let Some(s) = self.as_string() {
            s.starts_with("ERROR: ")
        } else {
            false
        }
    }
}

impl Default for JsValue {
    fn default() -> Self {
        JsValue::NULL
    }
}

// MessageChunk implementation for FormData
#[cfg(target_arch = "wasm32")]
impl MessageChunk for web_sys::FormData {
    fn bad_chunk(error: String) -> Self {
        let form_data = web_sys::FormData::new().unwrap_or_else(|_| {
            // Fallback if FormData::new() fails
            panic!("Failed to create FormData for error chunk: {}", error)
        });
        let _ = form_data.append_with_str("error", &error);
        form_data
    }

    fn error(&self) -> Option<&str> {
        // FormData doesn't have a direct way to check for errors
        // We check if it has an "error" field
        if let Ok(error_value) = self.get("error").as_string().ok_or("") {
            if !error_value.is_empty() {
                // This is a static string that we can't return a reference to
                // Return None since we can't provide a &str reference
                return None;
            }
        }
        None
    }

    fn is_error(&self) -> bool {
        // Check if FormData has an "error" field
        if let Ok(error_value) = self.get("error").as_string().ok_or("") {
            !error_value.is_empty()
        } else {
            false
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl MessageChunk for () {
    fn bad_chunk(_error: String) -> Self {
        ()
    }

    fn error(&self) -> Option<&str> {
        None
    }

    fn is_error(&self) -> bool {
        false
    }
}

pub mod body;
pub mod client;
pub mod request;
pub mod response;

// WASM-specific utilities
#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Object, Promise, Uint8Array};

/// Handle JavaScript errors in WASM context
#[cfg(target_arch = "wasm32")]
pub fn handle_error(error: JsValue) -> Error {
    if let Some(error_str) = error.as_string() {
        Error::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            error_str,
        ))
    } else {
        Error::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Unknown JavaScript error",
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn handle_error(_error: String) -> Error {
    Error::from(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "WASM functionality not available on this platform",
    ))
}

/// Convert JavaScript Promise to AsyncStream using callback pattern
#[cfg(target_arch = "wasm32")]
pub fn promise_to_stream(promise: Promise) -> fluent_ai_async::AsyncStream<JsValue, 1024> {
    use fluent_ai_async::{AsyncStream, emit};
    use wasm_bindgen::prelude::*;
    
    AsyncStream::with_channel(move |sender| {
        // Use JavaScript Promise.then() instead of Future-based approach
        let success_callback = Closure::once_into_js(move |js_val: JsValue| {
            emit!(sender, js_val);
        });
        
        let error_callback = Closure::once_into_js(move |error: JsValue| {
            emit!(sender, JsValue::bad_chunk(format!("Promise error: {:?}", error)));
        });
        
        // Use native JavaScript Promise.then() method
        let _ = promise.then2(&success_callback, &error_callback);
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn promise_to_stream(_promise: ()) -> fluent_ai_async::AsyncStream<JsValue, 1024> {
    use fluent_ai_async::{AsyncStream, emit};
    
    AsyncStream::with_channel(move |sender| {
        emit!(sender, JsValue::bad_chunk(
            "WASM functionality not available on this platform".to_string()
        ));
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
unsafe extern "C" {
    #[wasm_bindgen(js_name = "setTimeout")]
    fn set_timeout(handler: &Function, timeout: i32) -> JsValue;

    #[wasm_bindgen(js_name = "clearTimeout")]
    fn clear_timeout(handle: JsValue) -> JsValue;
}

fn promise<T>(
    promise: js_sys::Promise,
) -> fluent_ai_async::AsyncStream<T, 1024>
where
    T: JsCast + MessageChunk,
{
    use fluent_ai_async::{AsyncStream, emit};

    AsyncStream::with_channel(move |sender| {
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::prelude::*;
            
            // Use JavaScript Promise.then() instead of Future-based approach
            let success_callback = Closure::once_into_js(move |js_val: JsValue| {
                match wasm_bindgen::JsCast::dyn_into::<T>(js_val) {
                    Ok(result) => emit!(sender, result),
                    Err(_js_val) => {
                        emit!(sender, T::bad_chunk("promise resolved to unexpected type".to_string()))
                    }
                }
            });
            
            let error_callback = Closure::once_into_js(move |error: JsValue| {
                emit!(sender, T::bad_chunk(format!("WASM error: {:?}", error)))
            });
            
            // Use native JavaScript Promise.then() method
            let _ = promise.then2(&success_callback, &error_callback);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            emit!(sender, T::bad_chunk("WASM functionality not available on this platform".to_string()));
        }
    })
}

/// A guard that cancels a fetch request when dropped.
#[cfg(target_arch = "wasm32")]
struct AbortGuard {
    ctrl: AbortController,
    timeout: Option<(JsValue, Closure<dyn FnMut()>)>,
}

#[cfg(not(target_arch = "wasm32"))]
struct AbortGuard;

impl AbortGuard {
    fn new() -> std::result::Result<Self, crate::HttpError> {
        Ok(AbortGuard {
            ctrl: AbortController::new()
                .map_err(|e| crate::Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))
                .map_err(crate::client::core::ClientBuilder)?,
            timeout: None,
        })
    }

    fn signal(&self) -> AbortSignal {
        self.ctrl.signal()
    }

    #[cfg(target_arch = "wasm32")]
    fn timeout(&mut self, timeout: Duration) {
        let ctrl = self.ctrl.clone();
        let abort =
            Closure::once(move || ctrl.abort_with_reason(&"crate::client::HttpClient::errors::TimedOut".into()));
        let timeout = set_timeout(
            abort.as_ref().unchecked_ref::<js_sys::Function>(),
            match timeout.as_millis().try_into() {
                Ok(millis) => millis,
                Err(_) => return, // Skip if timeout conversion fails
            },
        );
        if let Some((id, _)) = self.timeout.replace((timeout, abort)) {
            clear_timeout(id);
        }
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        self.ctrl.abort();
        if let Some((id, _)) = self.timeout.take() {
            #[cfg(target_arch = "wasm32")]
            clear_timeout(id.as_f64().unwrap() as i32);
        }
    }
}
