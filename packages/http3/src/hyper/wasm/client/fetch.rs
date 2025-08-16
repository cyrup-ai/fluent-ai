//! WASM fetch implementation for HTTP requests

use std::convert::TryInto;
use std::fmt;

use bytes::Bytes;
use http::header::{HeaderMap, HeaderName, HeaderValue};
use http::{Method, Request, Response, StatusCode, Uri, Version};
use serde::{Deserialize, Serialize};
use serde_json;
use url::Url;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{Headers, RequestInit, RequestMode, Window};

#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Object, Uint8Array};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

use crate::hyper::error::{Error, Result};
use crate::hyper::Response as HttpResponse;
use crate::hyper::wasm::body::Body;
use crate::hyper::wasm::request::Request as WasmRequest;
// ResponseExt not available - removed
use crate::hyper::wasm::{handle_error, AbortController, AbortSignal};
use fluent_ai_async::AsyncStream;

#[cfg(target_arch = "wasm32")]
use web_sys::{Request as WebRequest, RequestCredentials, RequestRedirect};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
unsafe extern "C" {
    #[wasm_bindgen(js_name = fetch)]
    fn fetch_with_request(input: &web_sys::Request) -> Promise;
}

#[cfg(target_arch = "wasm32")]
fn js_fetch(req: &web_sys::Request) -> Promise {
    use wasm_bindgen::{JsCast, JsValue};
    let global = js_sys::global();

    if let Ok(true) = js_sys::Reflect::has(&global, &JsValue::from_str("ServiceWorkerGlobalScope"))
    {
        global
            .unchecked_into::<web_sys::ServiceWorkerGlobalScope>()
            .fetch_with_request(req)
    } else {
        // browser
        fetch_with_request(req)
    }
}

// Using current web-sys API for maximum compatibility, ignore their deprecation.
#[allow(deprecated)]
pub(super) fn fetch(req: Request<crate::hyper::Body>) -> AsyncStream<Response<crate::hyper::Body>> {
    AsyncStream::with_channel(move |sender| {
        wasm_bindgen_futures::spawn_local(async move {
            // Build the js Request
            let mut init = web_sys::RequestInit::new();
            init.method(req.method().as_str());

            // convert HeaderMap to Headers
            let js_headers = web_sys::Headers::new()
                .map_err(|e| Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))
                .map_err(crate::error::builder)?;

            for (name, value) in req.headers() {
                js_headers
                    .append(
                        name.as_str(),
                        value.to_str().map_err(crate::error::builder)?,
                    )
                    .map_err(|e| Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))
                    .map_err(crate::error::builder)?;
            }
            init.headers(&js_headers.into());

            // When req.cors is true, do nothing because the default mode is 'cors'
            if !req.cors {
                init.mode(web_sys::RequestMode::NoCors);
            }

            if let Some(creds) = req.credentials {
                init.credentials(creds);
            }

            if let Some(cache) = req.cache {
                init.set_cache(cache);
            }

            if let Some(body) = req.body() {
                if !body.is_empty() {
                    init.body(Some(body.to_js_value()?.as_ref()));
                }
            }

            #[cfg(target_arch = "wasm32")]
            let mut abort = {
                use web_sys::AbortController;
                let controller = AbortController::new().map_err(crate::error::wasm)?;
                controller
            };
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(_timeout) = req.timeout() {
                    // TODO: Implement timeout handling for WASM
                }
                init.signal(Some(&abort.signal()));
            }

            let js_req = web_sys::Request::new_with_str_and_init(req.url().as_str(), &init)
                .map_err(|e| Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))
                .map_err(crate::error::builder)?;

            // Await the fetch() promise
            #[cfg(target_arch = "wasm32")]
            let p = web_sys::window().unwrap().fetch_with_request(&js_req);
            #[cfg(target_arch = "wasm32")]
            let js_resp = {
                use wasm_bindgen_futures::JsFuture;
                JsFuture::from(p)
                    .await
                    .map_err(crate::error::wasm)?
                    .dyn_into::<web_sys::Response>()
                    .map_err(crate::error::wasm)?
            };

            #[cfg(target_arch = "wasm32")]
            let (mut resp, url, js_headers) = {
                // Convert from the js Response
                let mut resp = http::Response::builder().status(js_resp.status());
                let url = Url::parse(&js_resp.url()).expect_throw("url parse");
                let js_headers = js_resp.headers();
                (resp, url, js_headers)
            };

            #[cfg(target_arch = "wasm32")]
            {
                let js_iter = js_sys::try_iter(&js_headers)
                    .expect_throw("headers try_iter")
                    .expect_throw("headers have an iterator");

                for item in js_iter {
                    let item = item.expect_throw("headers iterator doesn't throw");
                    let serialized_headers: String = js_sys::JSON::stringify(&item)
                        .expect_throw("serialized headers")
                        .into();
                    let [name, value]: [String; 2] = serde_json::from_str(&serialized_headers)
                        .expect_throw("deserializable serialized headers");
                    resp = resp.header(&name, &value);
                }

                // Complete the response building and create our Response type
                let http_response = resp.body(js_resp).map_err(crate::error::decode)?;
                let response = Response::new(http_response, url, abort);
                fluent_ai_async::emit!(sender, Ok(response));
            }
        });
    })
}
