#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "multipart")]
use super::super::multipart::Form;
use super::types::{Body, Inner, Single};

impl Body {
    /// Returns a reference to the internal data of the `Body`.
    ///
    /// `None` is returned, if the underlying data is a multipart form.
    #[inline]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.inner {
            Inner::Single(single) => Some(single.as_bytes()),
            #[cfg(feature = "multipart")]
            Inner::MultipartForm(_) => None,
        }
    }

    pub(crate) fn to_js_value(&self) -> crate::Result<JsValue> {
        match &self.inner {
            Inner::Single(single) => Ok(single.to_js_value()),
            #[cfg(feature = "multipart")]
            Inner::MultipartForm(form) => {
                let form_data = form.to_form_data()?;
                let js_value: &JsValue = form_data.as_ref();
                Ok(js_value.to_owned())
            }
        }
    }

    #[cfg(feature = "multipart")]
    pub(crate) fn as_single(&self) -> Option<&Single> {
        match &self.inner {
            Inner::Single(single) => Some(single),
            Inner::MultipartForm(_) => None,
        }
    }

    #[inline]
    #[cfg(feature = "multipart")]
    pub(crate) fn from_form(f: Form) -> Body {
        Self {
            inner: Inner::MultipartForm(f),
        }
    }

    /// into_part turns a regular body into the body of a multipart/form-data part.
    #[cfg(feature = "multipart")]
    pub(crate) fn into_part(self) -> Body {
        match self.inner {
            Inner::Single(single) => Self {
                inner: Inner::Single(single),
            },
            Inner::MultipartForm(form) => Self {
                inner: Inner::MultipartForm(form),
            },
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match &self.inner {
            Inner::Single(single) => single.is_empty(),
            #[cfg(feature = "multipart")]
            Inner::MultipartForm(form) => form.is_empty(),
        }
    }

    pub(crate) fn try_clone(&self) -> Option<Body> {
        match &self.inner {
            Inner::Single(single) => Some(Self {
                inner: Inner::Single(single.clone()),
            }),
            #[cfg(feature = "multipart")]
            Inner::MultipartForm(_) => None,
        }
    }
}
