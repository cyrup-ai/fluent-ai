//! WASM fetch implementation for HTTP requests

use std::convert::TryInto;

use fluent_ai_async::AsyncStream;
use js_sys::{JSON, Promise};
use url::Url;
use wasm_bindgen::prelude::{UnwrapThrowExt as _, wasm_bindgen};

use super::{AbortGuard, Request, Response};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = fetch)]
    fn fetch_with_request(input: &web_sys::Request) -> Promise;
}

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
pub(super) fn fetch(req: Request) -> AsyncStream<Result<Response, crate::Error>> {
    AsyncStream::with_channel(move |sender| {
        wasm_bindgen_futures::spawn_local(async move {
            // Build the js Request
            let mut init = web_sys::RequestInit::new();
            init.method(req.method().as_str());

            // convert HeaderMap to Headers
            let js_headers = web_sys::Headers::new()
                .map_err(crate::error::wasm)
                .map_err(crate::error::builder)?;

            for (name, value) in req.headers() {
                js_headers
                    .append(
                        name.as_str(),
                        value.to_str().map_err(crate::error::builder)?,
                    )
                    .map_err(crate::error::wasm)
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

            let mut abort = AbortGuard::new()?;
            if let Some(timeout) = req.timeout() {
                abort.timeout(*timeout);
            }
            init.signal(Some(&abort.signal()));

            let js_req = web_sys::Request::new_with_str_and_init(req.url().as_str(), &init)
                .map_err(crate::error::wasm)
                .map_err(crate::error::builder)?;

            // Await the fetch() promise
            let p = js_fetch(&js_req);
            let js_resp = super::promise::<web_sys::Response>(p)
                .await
                .map_err(|error| {
                    if error.to_string() == "JsValue(\"crate::hyper::errors::TimedOut\")" {
                        crate::error::TimedOut.into()
                    } else {
                        error
                    }
                })
                .map_err(crate::error::request)?;

            // Convert from the js Response
            let mut resp = http::Response::builder().status(js_resp.status());

            let url = Url::parse(&js_resp.url()).expect_throw("url parse");

            let js_headers = js_resp.headers();
            let js_iter = js_sys::try_iter(&js_headers)
                .expect_throw("headers try_iter")
                .expect_throw("headers have an iterator");

            for item in js_iter {
                let item = item.expect_throw("headers iterator doesn't throw");
                let serialized_headers: String = JSON::stringify(&item)
                    .expect_throw("serialized headers")
                    .into();
                let [name, value]: [String; 2] = serde_json::from_str(&serialized_headers)
                    .expect_throw("deserializable serialized headers");
                resp = resp.header(&name, &value);
            }

            // Complete the response building and create our Response type
            let http_response = resp.body(js_resp).map_err(crate::error::builder)?;

            let response = Response::new(http_response, url, abort);
            fluent_ai_async::emit!(sender, Ok(response));
        });
    })
}
