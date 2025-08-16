//! Cookie configuration methods for ClientBuilder
//!
//! Contains methods for configuring persistent cookie stores
//! and cookie handling behavior.

use std::sync::Arc;

use super::types::ClientBuilder;
#[cfg(feature = "cookies")]
use crate::hyper::cookie;

impl ClientBuilder {
    /// Enable a persistent cookie store for the client.
    ///
    /// Cookies received in responses will be preserved and included in
    /// additional requests.
    ///
    /// By default, no cookie store is used. Enabling the cookie store
    /// with `cookie_store(true)` will set the store to a default implementation.
    /// It is **not** necessary to call [cookie_store(true)](crate::ClientBuilder::cookie_store) if [cookie_provider(my_cookie_store)](crate::ClientBuilder::cookie_provider)
    /// is used; calling [cookie_store(true)](crate::ClientBuilder::cookie_store) _after_ [cookie_provider(my_cookie_store)](crate::ClientBuilder::cookie_provider) will result
    /// in the provided `my_cookie_store` being **overridden** with a default implementation.
    ///
    /// # Optional
    ///
    /// This requires the optional `cookies` feature to be enabled.
    #[cfg(feature = "cookies")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cookies")))]
    pub fn cookie_store(mut self, enable: bool) -> ClientBuilder {
        if enable {
            self.cookie_provider(Arc::new(cookie::Jar::default()))
        } else {
            self.config.cookie_store = None;
            self
        }
    }

    /// Set the persistent cookie store for the client.
    ///
    /// Cookies received in responses will be passed to this store, and
    /// additional requests will query this store for cookies.
    ///
    /// By default, no cookie store is used. It is **not** necessary to also call
    /// [cookie_store(true)](crate::ClientBuilder::cookie_store) if [cookie_provider(my_cookie_store)](crate::ClientBuilder::cookie_provider) is used; calling
    /// [cookie_store(true)](crate::ClientBuilder::cookie_store) _after_ [cookie_provider(my_cookie_store)](crate::ClientBuilder::cookie_provider) will result
    /// in the provided `my_cookie_store` being **overridden** with a default implementation.
    ///
    /// # Optional
    ///
    /// This requires the optional `cookies` feature to be enabled.
    #[cfg(feature = "cookies")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cookies")))]
    pub fn cookie_provider<C: cookie::CookieStore + 'static>(
        mut self,
        cookie_store: Arc<C>,
    ) -> ClientBuilder {
        self.config.cookie_store = Some(cookie_store as _);
        self
    }
}
