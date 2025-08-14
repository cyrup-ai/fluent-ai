#![cfg(feature = "http3")]

pub(crate) mod connect;
pub(crate) mod dns;
mod pool;

use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
#[cfg(feature = "cookies")]
use std::sync::Arc;
use std::time::Duration;

use connect::H3Connector;
use http::{Request, Response};
use log::trace;
use pool::{Key, Pool, PoolClient};

use super::body::ResponseBody;
#[cfg(feature = "cookies")]
use crate::cookie;
use crate::hyper::error::{BoxError, Error, Kind};
use crate::{Body, error};

#[derive(Clone)]
pub(crate) struct H3Client {
    pool: Pool,
    connector: H3Connector,
    #[cfg(feature = "cookies")]
    cookie_store: Option<Arc<dyn cookie::CookieStore>>,
}

impl H3Client {
    #[cfg(not(feature = "cookies"))]
    pub fn new(connector: H3Connector, pool_timeout: Option<Duration>) -> Self {
        H3Client {
            pool: Pool::new(pool_timeout),
            connector,
        }
    }

    #[cfg(feature = "cookies")]
    pub fn new(
        connector: H3Connector,
        pool_timeout: Option<Duration>,
        cookie_store: Option<Arc<dyn cookie::CookieStore>>,
    ) -> Self {
        H3Client {
            pool: Pool::new(pool_timeout),
            connector,
            cookie_store,
        }
    }

    fn get_pooled_client(&mut self, key: Key) -> fluent_ai_async::AsyncStream<Result<PoolClient, BoxError>> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let mut pool = self.pool.clone();
        let mut connector = self.connector.clone();
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<PoolClient, BoxError> {
                if let Some(client) = pool.try_pool(&key) {
                    trace!("getting client from pool with key {key:?}");
                    return Ok(client);
                }

                trace!("did not find connection {key:?} in pool so connecting...");

                let dest = pool::domain_as_uri(key.clone());

                let lock = match pool.connecting(&key) {
                    pool::Connecting::InProgress(waiter) => {
                        trace!("connecting to {key:?} is already in progress, subscribing...");

                        let mut receive_stream = waiter.receive();
                        match receive_stream.try_next() {
                            Some(Some(client)) => {
                                return Ok(client);
                            },
                            _ => {
                                return Err("failed to establish connection for HTTP/3 request".into());
                            },
                        }
                    }
                    pool::Connecting::Acquired(lock) => lock,
                };
                trace!("connecting to {key:?}...");
                let mut connect_stream = connector.connect(dest);
                let (driver, tx) = match connect_stream.try_next() {
                    Some(Ok(result)) => result,
                    Some(Err(e)) => {
                        return Err(format!("get pooled client connection failed: {}", e).into());
                    },
                    None => {
                        return Err("connection failed".into());
                    },
                };
                trace!("saving new pooled connection to {key:?}");
                let client = pool.new_connection(lock, driver, tx);
                Ok(client)
            });
            
            match task.collect() {
                Ok(pooled_client) => emit!(sender, Ok(pooled_client)),
                Err(e) => emit!(sender, Err(format!("get pooled client task failed: {}", e).into())),
            }
        })
    }

    #[cfg(not(feature = "cookies"))]
    fn send_request(
        mut self,
        key: Key,
        req: Request<Body>,
    ) -> fluent_ai_async::AsyncStream<Response<ResponseBody>> {
        fluent_ai_async::AsyncStream::with_channel(move |sender| {
            let task = fluent_ai_async::spawn_task(move || -> Result<Response<ResponseBody>, Error> {
                let mut pooled_stream = self.get_pooled_client(key);
                let mut pooled = match pooled_stream.try_next() {
                    Some(Ok(client)) => client,
                    Some(Err(e)) => {
                        return Err(Error::new(Kind::Request, Some(error::request(e))));
                    },
                    None => {
                        return Err(Error::new(Kind::Request, None::<Error>));
                    },
                };
                let mut request_stream = pooled.send_request(req);
                match request_stream.try_next() {
                    Some(result) => match result {
                        Ok(response) => Ok(response),
                        Err(e) => Err(Error::new(Kind::Request, Some(error::request(e)))),
                    },
                    None => {
                        Err(Error::new(Kind::Request, None::<Error>))
                    },
                }
            });

            match task.collect() {
                Ok(response) => fluent_ai_async::emit!(sender, response),
                Err(e) => fluent_ai_async::handle_error!(Error::new(Kind::Request, Some(e)), "h3 send request task"),
            }
        })
    }

    #[cfg(feature = "cookies")]
    fn send_request(
        mut self,
        key: Key,
        mut req: Request<Body>,
    ) -> fluent_ai_async::AsyncStream<Response<ResponseBody>> {
        fluent_ai_async::AsyncStream::with_channel(move |sender| {
            let task = fluent_ai_async::spawn_task(move || -> Result<Response<ResponseBody>, Error> {
                let mut pooled_stream = self.get_pooled_client(key);
                let mut pooled = match pooled_stream.try_next() {
                    Some(Ok(client)) => client,
                    Some(Err(e)) => {
                        return Err(Error::new(Kind::Request, Some(error::request(e))));
                    },
                    None => {
                        return Err(Error::new(Kind::Request, None::<Error>));
                    },
                };

                let url = match url::Url::parse(req.uri().to_string().as_str()) {
                    Ok(url) => url,
                    Err(e) => {
                        return Err(Error::new(Kind::Request, Some(error::request(e))));
                    },
                };
                if let Some(cookie_store) = self.cookie_store.as_ref() {
                    if req.headers().get(crate::header::COOKIE).is_none() {
                        let headers = req.headers_mut();
                        crate::util::add_cookie_header(headers, &**cookie_store, &url);
                    }
                }

                let mut request_stream = pooled.send_request(req);
                let res = match request_stream.try_next() {
                    Some(response) => response,
                    None => {
                        return Err(Error::new(Kind::Request, None::<Error>));
                    },
                };

                if let Some(ref cookie_store) = self.cookie_store {
                    let mut cookies = cookie::extract_response_cookie_headers(res.headers()).peekable();
                    if cookies.peek().is_some() {
                        cookie_store.set_cookies(&mut cookies, &url);
                    }
                }

                Ok(res)
            });

            match task.collect() {
                Ok(result) => match result {
                    Ok(response) => fluent_ai_async::emit!(sender, response),
                    Err(e) => fluent_ai_async::handle_error!(e, "h3 send request with cookies"),
                },
                Err(e) => fluent_ai_async::handle_error!(Error::new(Kind::Request, Some(e)), "h3 send request with cookies task"),
            }
        })
    }

    pub fn request(&self, mut req: Request<Body>) -> crate::hyper::StreamFuture<Result<Response<ResponseBody>, Error>> {
        let pool_key = match pool::extract_domain(req.uri_mut()) {
            Ok(s) => s,
            Err(e) => {
                let error_stream = fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, Err(e));
                });
                return crate::hyper::StreamFuture::new(error_stream);
            }
        };
        let response_stream = execute_h3_request_stream(self.clone(), pool_key, req);
        crate::hyper::StreamFuture::new(response_stream)
    }
}

impl Service<Request<Body>> for H3Client {
    type Response = Response<ResponseBody>;
    type Error = Error;
    type Future = crate::hyper::StreamFuture<Result<Response<ResponseBody>, Error>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let mut req_clone = req;
        let pool_key = match pool::extract_domain(req_clone.uri_mut()) {
            Ok(s) => s,
            Err(e) => {
                let error_stream = fluent_ai_async::AsyncStream::with_channel(move |sender| {
                    fluent_ai_async::emit!(sender, Err(e));
                });
                return crate::hyper::StreamFuture::new(error_stream);
            }
        };
        
        let response_stream = execute_h3_request_stream(self.clone(), pool_key, req_clone);
        crate::hyper::StreamFuture::new(response_stream)
    }
}


