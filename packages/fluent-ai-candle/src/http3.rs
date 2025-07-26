//! Candle HTTP3 - Re-export fluent_ai_http3 with Candle prefixes

// Re-export fluent_ai_http3 types with Candle prefixes
pub use fluent_ai_http3::{
    HttpClient as CandleHttpClient,
    HttpConfig as CandleHttpConfig,
    HttpRequest as CandleHttpRequest,
    HttpResponse as CandleHttpResponse,
    HttpError as CandleHttpError,
    Http3 as CandleHttp3,
    HttpChunk as CandleHttpChunk,
}