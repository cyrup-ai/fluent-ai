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
use web_sys::{AbortController, AbortSignal};

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

#[wasm_bindgen]
unsafe extern "C" {
    #[wasm_bindgen(js_name = "setTimeout")]
    fn set_timeout(handler: &Function, timeout: i32) -> JsValue;

    #[wasm_bindgen(js_name = "clearTimeout")]
    fn clear_timeout(handle: JsValue) -> JsValue;
}

fn promise<T>(
    promise: js_sys::Promise,
) -> fluent_ai_async::AsyncStream<Result<T, crate::error::BoxError>>
where
    T: JsCast,
{
    use fluent_ai_async::{AsyncStream, emit, handle_error};
    use wasm_bindgen_futures::JsFuture;

    AsyncStream::with_channel(move |sender| {
        wasm_bindgen_futures::spawn_local(async move {
            match JsFuture::from(promise).await {
                Ok(js_val) => match js_val.dyn_into::<T>() {
                    Ok(result) => emit!(sender, Ok(result)),
                    Err(_js_val) => {
                        emit!(sender, Err("promise resolved to unexpected type".into()))
                    }
                },
                Err(e) => emit!(sender, Err(crate::error::wasm(e))),
            }
        });
    })
}

/// A guard that cancels a fetch request when dropped.
struct AbortGuard {
    ctrl: AbortController,
    timeout: Option<(JsValue, Closure<dyn FnMut()>)>,
}

impl AbortGuard {
    fn new() -> crate::Result<Self> {
        Ok(AbortGuard {
            ctrl: AbortController::new()
                .map_err(crate::error::wasm)
                .map_err(crate::error::builder)?,
            timeout: None,
        })
    }

    fn signal(&self) -> AbortSignal {
        self.ctrl.signal()
    }

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
            clear_timeout(id);
        }
    }
}
