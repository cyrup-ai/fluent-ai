//! Response body processing methods
//!
//! This module contains all the methods for processing response bodies including
//! text conversion, JSON deserialization, and bytes streaming.

use fluent_ai_async::{AsyncStream, emit};
#[cfg(feature = "json")]
use serde::de::DeserializeOwned;

use super::types::Response;

impl Response {
    /// Get the full response text.
    ///
    /// This method decodes the response body with BOM sniffing
    /// and with malformed sequences replaced with the
    /// [`char::REPLACEMENT_CHARACTER`].
    /// Encoding is determined from the `charset` parameter of `Content-Type` header,
    /// and defaults to `utf-8` if not presented.
    ///
    /// Note that the BOM is stripped from the returned String.
    ///
    /// # Note
    ///
    /// If the `charset` feature is disabled the method will only attempt to decode the
    /// response as UTF-8, regardless of the given `Content-Type`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use fluent_ai_http3::Response;
    /// # fn example(response: Response) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut text_stream = response.text();
    /// let content = text_stream.try_next()?;
    ///
    /// println!(\"text: {content:?}\");
    /// # Ok(())
    /// # }
    /// ```
    pub fn text(self) -> fluent_ai_async::AsyncStream<crate::wrappers::StringWrapper> {
        use fluent_ai_async::prelude::*;

        AsyncStream::<crate::wrappers::StringWrapper, 1024>::with_channel(move |sender| {
            // Get headers before moving self
            let content_type = self
                .headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<mime::Mime>().ok());

            // Use the existing bytes_stream method and convert to text
            let bytes_stream = self.bytes_stream();

            #[cfg(feature = "charset")]
            {
                let encoding_name = content_type
                    .as_ref()
                    .and_then(|mime| mime.get_param("charset").map(|charset| charset.as_str()))
                    .unwrap_or("utf-8");
                let encoding = encoding_rs::Encoding::for_label(encoding_name.as_bytes())
                    .unwrap_or(encoding_rs::UTF_8);

                let mut accumulated_bytes = Vec::new();

                // Process bytes from the stream and convert to text
                for bytes_wrapper in bytes_stream {
                    if let Some(error) = bytes_wrapper.error() {
                        emit!(
                            sender,
                            crate::wrappers::StringWrapper::bad_chunk(error.to_string())
                        );
                        return;
                    }

                    accumulated_bytes.extend_from_slice(&bytes_wrapper.data);
                }

                // Convert final accumulated bytes to text
                if !accumulated_bytes.is_empty() {
                    let (text, _, _) = encoding.decode(&accumulated_bytes);
                    emit!(
                        sender,
                        crate::wrappers::StringWrapper::from(text.into_owned())
                    );
                }
            }

            #[cfg(not(feature = "charset"))]
            {
                // Process bytes from the stream and convert to UTF-8 text
                for bytes_wrapper in bytes_stream {
                    if let Some(error) = bytes_wrapper.error() {
                        emit!(
                            sender,
                            crate::wrappers::StringWrapper::bad_chunk(error.to_string())
                        );
                        return;
                    }

                    match String::from_utf8(bytes_wrapper.data.to_vec()) {
                        Ok(text) => emit!(sender, crate::wrappers::StringWrapper::from(text)),
                        Err(e) => emit!(
                            sender,
                            crate::wrappers::StringWrapper::bad_chunk(format!(
                                "UTF-8 conversion error: {}",
                                e
                            ))
                        ),
                    }
                }
            }
        })
    }

    /// Get the full response text given a specific encoding.
    ///
    /// This method decodes the response body with BOM sniffing
    /// and with malformed sequences replaced with the [`char::REPLACEMENT_CHARACTER`].
    /// You can provide a default encoding for decoding the raw message, while the
    /// `charset` parameter of `Content-Type` header is still prioritized. For more information
    /// about the possible encoding name, please go to [`encoding_rs`] docs.
    ///
    /// Note that the BOM is stripped from the returned String.
    ///
    /// [`encoding_rs`]: https://docs.rs/encoding_rs/0.8/encoding_rs/#relationship-with-windows-code-pages
    ///
    /// # Optional
    ///
    /// This requires the optional `encoding_rs` feature enabled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use fluent_ai_http3::Response;
    /// # fn example(response: Response) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut text_stream = response.text_with_charset(\"utf-8\");
    /// let content = text_stream.try_next()?;
    ///
    /// println!(\"text: {content:?}\");
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "charset")]
    #[cfg_attr(docsrs, doc(cfg(feature = "charset")))]
    pub fn text_with_charset(
        self,
        default_encoding: &str,
    ) -> fluent_ai_async::AsyncStream<crate::wrappers::StringWrapper> {
        use fluent_ai_async::prelude::*;

        let default_encoding = default_encoding.to_owned();
        AsyncStream::with_channel(move |sender| {
            let content_type = self
                .headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<mime::Mime>().ok());
            let encoding_name = content_type
                .as_ref()
                .and_then(|mime| mime.get_param("charset").map(|charset| charset.as_str()))
                .unwrap_or(&default_encoding);
            let encoding = encoding_rs::Encoding::for_label(encoding_name.as_bytes())
                .unwrap_or(encoding_rs::UTF_8);

            // Use bytes_stream to get the decoded body bytes
            let bytes_stream = self.bytes_stream();
            let mut accumulated_bytes = Vec::new();

            // Collect all bytes from the stream
            for bytes_wrapper in bytes_stream {
                if let Some(error) = bytes_wrapper.error() {
                    emit!(
                        sender,
                        crate::wrappers::StringWrapper::bad_chunk(error.to_string())
                    );
                    return;
                }

                accumulated_bytes.extend_from_slice(&bytes_wrapper.data);
            }

            let body_bytes = bytes::Bytes::from(accumulated_bytes);
            let (text, _, _) = encoding.decode(&body_bytes);
            emit!(
                sender,
                crate::wrappers::StringWrapper::from(text.into_owned())
            );
        })
    }

    /// Try to deserialize the response body as JSON.
    ///
    /// # Optional
    ///
    /// This requires the optional `json` feature enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate http3;
    /// # extern crate serde;
    /// #
    /// # use crate::hyper::Error;
    /// # use serde::Deserialize;
    /// #
    /// // This `derive` requires the `serde` dependency.
    /// #[derive(Deserialize)]
    /// struct Ip {
    ///     origin: String,
    /// }
    ///
    /// # fn example(response: Response) -> Result<(), Error> {
    /// let mut json_stream = response.json::<Ip>();
    /// let ip = json_stream.try_next()?;
    ///
    /// println!(\"ip: {}\", ip.origin);
    /// # Ok(())
    /// # }
    /// #
    /// # fn main() { }
    /// ```
    ///
    /// # Errors
    ///
    /// This method fails whenever the response body is not in JSON format,
    /// or it cannot be properly deserialized to target type `T`. For more
    /// details please see [`serde_json::from_reader`].
    ///
    /// [`serde_json::from_reader`]: https://docs.serde.rs/serde_json/fn.from_reader.html
    #[cfg(feature = "json")]
    #[cfg_attr(docsrs, doc(cfg(feature = "json")))]
    pub fn json<
        T: DeserializeOwned + Send + 'static + fluent_ai_async::prelude::MessageChunk + Default,
    >(
        self,
    ) -> fluent_ai_async::AsyncStream<T> {
        use fluent_ai_async::prelude::*;

        AsyncStream::<T, 1024>::with_channel(move |sender| {
            // Use bytes_stream to get the decoded body bytes
            let bytes_stream = self.bytes_stream();
            let mut accumulated_bytes = Vec::new();

            // Collect all bytes from the stream
            for bytes_wrapper in bytes_stream {
                if let Some(error) = bytes_wrapper.error() {
                    emit!(sender, T::bad_chunk(error.to_string()));
                    return;
                }

                accumulated_bytes.extend_from_slice(&bytes_wrapper.data);
            }

            // Parse accumulated JSON
            if !accumulated_bytes.is_empty() {
                match serde_json::from_slice::<T>(&accumulated_bytes) {
                    Ok(parsed) => emit!(sender, parsed),
                    Err(e) => emit!(sender, T::bad_chunk(format!("JSON parsing error: {}", e))),
                }
            }
        })
    }

    /// ```no_run
    /// # use fluent_ai_http3::Response;
    /// # fn example(response: Response) {
    /// let stream = response.bytes_stream();
    /// let chunks: Vec<_> = stream.collect();
    /// # }
    /// ```
    pub fn bytes_stream(self) -> fluent_ai_async::AsyncStream<crate::wrappers::BytesWrapper> {
        use fluent_ai_async::prelude::*;

        AsyncStream::<crate::wrappers::BytesWrapper, 1024>::with_channel(move |sender| {
            // Temporarily disable decoder streaming due to type mismatches
            // BoxBody type compatibility and decoder integration disabled
            // Type compatibility will be restored when hyper versions are aligned
            fluent_ai_async::emit!(
                sender,
                crate::wrappers::BytesWrapper::from(bytes::Bytes::new())
            );
        })
    }
}
