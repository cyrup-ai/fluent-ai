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
use web_sys::{Headers, RequestInit, RequestMode, Window};

#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Object, Uint8Array};

use crate::error::{Error, Result};
use crate::prelude::HttpResponse;
use crate::wasm::body::Body;
use crate::wasm::request::Request as WasmRequest;
// ResponseExt not available - removed
use crate::wasm::{handle_error, AbortController, AbortSignal};
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
pub(super) fn fetch(req: Request<crate::wasm::body::Body>) -> AsyncStream<Response<crate::wasm::body::Body>, 1024> {
    use fluent_ai_async::emit;
    
    AsyncStream::with_channel(move |sender| {
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::prelude::*;
            use wasm_bindgen::JsCast;
            
            // Build the request setup 
            let setup_result = (|| -> Result<(web_sys::Request, web_sys::AbortController), Error> {
                let mut init = web_sys::RequestInit::new();
                init.method(req.method().as_str());

                // convert HeaderMap to Headers
                let js_headers = web_sys::Headers::new()
                    .map_err(|e| Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))?;

                for (name, value) in req.headers() {
                    js_headers
                        .append(
                            name.as_str(),
                            value.to_str().map_err(crate::error::builder)?,
                        )
                        .map_err(|e| Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))?;
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

                let abort_controller = web_sys::AbortController::new()
                    .map_err(|e| Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))?;
                
                if let Some(_timeout) = req.timeout() {
                    // TODO: Implement timeout handling for WASM
                }
                init.signal(Some(&abort_controller.signal()));

                let js_req = web_sys::Request::new_with_str_and_init(req.url().as_str(), &init)
                    .map_err(|e| Error::from(std::io::Error::new(std::io::ErrorKind::Other, format!("WASM error: {:?}", e))))?;

                Ok((js_req, abort_controller))
            })();

            match setup_result {
                Ok((js_req, abort_controller)) => {
                    // Use fetch with promise callbacks instead of Future
                    let fetch_promise = web_sys::window().unwrap().fetch_with_request(&js_req);
                    
                    let success_callback = Closure::once_into_js(move |js_resp: JsValue| {
                        let web_response: web_sys::Response = js_resp.dyn_into().unwrap();
                        
                        // Convert from the js Response
                        let mut resp = http::Response::builder().status(web_response.status());
                        let url = Url::parse(&web_response.url()).expect("url parse");
                        let js_headers = web_response.headers();

                        let js_iter = js_sys::try_iter(&js_headers)
                            .expect("headers try_iter")
                            .expect("headers have an iterator");

                        for item in js_iter {
                            let item = item.expect("headers iterator doesn't throw");
                            let serialized_headers: String = js_sys::JSON::stringify(&item)
                                .expect("serialized headers")
                                .into();
                            let [name, value]: [String; 2] = serde_json::from_str(&serialized_headers)
                                .expect("deserializable serialized headers");
                            resp = resp.header(&name, &value);
                        }

                        // Complete the response building and create our Response type
                        match resp.body(web_response) {
                            Ok(http_response) => {
                                let response = Response::new(http_response, url, abort_controller);
                                emit!(sender, Ok(response));
                            }
                            Err(e) => {
                                emit!(sender, Err(Error::from(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    format!("Response build error: {}", e)
                                ))));
                            }
                        }
                    });
                    
                    let error_callback = Closure::once_into_js(move |error: JsValue| {
                        emit!(sender, Err(Error::from(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Fetch error: {:?}", error)
                        ))));
                    });
                    
                    // Use native JavaScript Promise.then() method
                    let _ = fetch_promise.then2(&success_callback, &error_callback);
                }
                Err(setup_error) => {
                    emit!(sender, Err(setup_error));
                }
            }
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            emit!(sender, Err(Error::from(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "WASM functionality not available on this platform"
            ))));
        }
    })
}
