//! Service interface implementation with AsyncStreamService trait
//!
//! Provides the main connect() method and AsyncStreamService trait implementation
//! with elite polling patterns and connection result wrapping.

use fluent_ai_async::{AsyncStream, emit, spawn_task};
use http::Uri;

use super::super::types::TcpStreamWrapper;
use super::core::ConnectorService;
use crate::hyper::async_stream_service::{AsyncStreamService, ConnResult};
use crate::hyper::error::BoxError;

impl ConnectorService {
    /// Direct connection method - replaces Service::call with AsyncStream
    pub fn connect(&mut self, dst: Uri) -> AsyncStream<TcpStreamWrapper> {
        let connector_service = self.clone();

        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                let mut connection_stream =
                    if let Some(_proxy) = connector_service.intercepted.matching(&dst) {
                        connector_service.connect_via_proxy(dst, "proxy")
                    } else {
                        connector_service.connect_with_maybe_proxy(dst, false)
                    };

                // Elite polling pattern - non-blocking stream consumption
                match connection_stream.try_next() {
                    Some(conn) => {
                        emit!(sender, conn);
                    }
                    None => {
                        emit!(
                            sender,
                            TcpStreamWrapper::bad_chunk(
                                "Connection stream ended without producing connection".to_string()
                            )
                        );
                    }
                }
            });
        })
    }
}

// AsyncStreamService implementation for ConnectorService
impl AsyncStreamService<Uri> for ConnectorService {
    type Response = TcpStreamWrapper;
    type Error = BoxError;

    fn is_ready(&mut self) -> bool {
        // ConnectorService is always ready to accept connections
        true
    }

    fn call(&mut self, request: Uri) -> AsyncStream<ConnResult<TcpStreamWrapper>> {
        // Convert the direct connect() result to the required ConnResult stream format
        let connection_stream = self.connect(request);

        AsyncStream::with_channel(move |sender| {
            spawn_task(move || match connection_stream.try_next() {
                Some(conn) => {
                    emit!(sender, ConnResult::success(conn));
                }
                None => {
                    emit!(sender, ConnResult::error("Connection establishment failed"));
                }
            });
        })
    }
}
