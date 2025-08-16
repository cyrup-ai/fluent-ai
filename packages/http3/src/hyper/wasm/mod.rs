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
use wasm_bindgen_futures::{JsFuture, spawn_local};
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
use std::future::Future;
#[cfg(target_arch = "wasm32")]
use std::pin::Pin;
#[cfg(target_arch = "wasm32")]
use std::task::{Context, Poll};

#[cfg(target_arch = "wasm32")]
use bytes::Bytes;
#[cfg(target_arch = "wasm32")]
use http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
#[cfg(target_arch = "wasm32")]
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use url::Url;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

use crate::hyper::error::{Error, Result};

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

/// Convert JavaScript Promise to Rust Future
#[cfg(target_arch = "wasm32")]
pub fn promise_to_future(promise: Promise) -> impl Future<Output = Result<JsValue>> {
    async move {
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(handle_error)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn promise_to_future(_promise: ()) -> impl Future<Output = Result<()>> {
    async move {
        Err(Error::from(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "WASM functionality not available on this platform",
        )))
    }
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
) -> fluent_ai_async::AsyncStream<T>
where
    T: JsCast,
{
    use fluent_ai_async::{AsyncStream, emit};


    AsyncStream::with_channel(move |sender| {
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen_futures::{spawn_local, JsFuture};
            spawn_local(async move {
                match JsFuture::from(promise).await {
                Ok(js_val) => match wasm_bindgen::JsCast::dyn_into::<T>(js_val) {
                    Ok(result) => emit!(sender, Ok(result)),
                    Err(_js_val) => {
                        emit!(sender, Err("promise resolved to unexpected type".into()))
                    }
                },
                Err(e) => emit!(sender, Err(crate::Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))),
            }
            });
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
    fn new() -> crate::Result<Self> {
        Ok(AbortGuard {
            ctrl: AbortController::new()
                .map_err(|e| crate::Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))
                .map_err(crate::error::builder)?,
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
            Closure::once(move || ctrl.abort_with_reason(&"crate::hyper::errors::TimedOut".into()));
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
