use std::borrow::Cow;

use bytes::Bytes;
use js_sys::Uint8Array;
use wasm_bindgen::JsValue;

use super::types::Single;

impl Single {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            Single::Bytes(bytes) => bytes.as_ref(),
            Single::Text(text) => text.as_bytes(),
        }
    }

    pub(crate) fn to_js_value(&self) -> JsValue {
        match self {
            Single::Bytes(bytes) => {
                let body_bytes: &[u8] = bytes.as_ref();
                let body_uint8_array: Uint8Array = body_bytes.into();
                let js_value: &JsValue = body_uint8_array.as_ref();
                js_value.to_owned()
            }
            Single::Text(text) => JsValue::from_str(text),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Single::Bytes(bytes) => bytes.is_empty(),
            Single::Text(text) => text.is_empty(),
        }
    }
}
