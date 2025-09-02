# HTTP3 Public API Documentation

## Original CORRECT API from Revision a97141b vs Current Implementation Status

This document provides a comprehensive comparison between the original CORRECT public API that should NEVER have been changed (revision a97141b) and the current broken implementation status.

**SOURCE OF TRUTH**: Revision a97141b - the last known good API

---

## 1. CORE BUILDER METHODS (Http3Builder)

### Constructor Methods (Http3Builder<BodyNotSet>)

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `new()` | `pub fn new(client: &HttpClient) -> Self` | 🔍 | | |
| `json()` | `pub fn json() -> Self` | ✅ EXISTS | `src/builder/builder_core.rs:117` | Creates JSON builder |
| `form_urlencoded()` | `pub fn form_urlencoded() -> Self` | ✅ EXISTS | `src/builder/builder_core.rs:126` | Creates form builder |
| `array_stream()` | `pub fn array_stream(self, jsonpath: &str) -> Http3Builder<JsonPathStreaming>` | 🔍 | | JSONPath streaming |

### Configuration Methods (Http3Builder<S>)

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `debug()` | `pub fn debug(mut self) -> Self` | 🔍 | | Enable debug logging |
| `url()` | `pub fn url(mut self, url: &str) -> Self` | 🔍 | | Set target URL |
| `content_type()` | `pub fn content_type(self, content_type: ContentType) -> Self` | 🔍 | | Set content type |
| `timeout_seconds()` | `pub fn timeout_seconds(mut self, seconds: u64) -> Self` | 🔍 | | Set timeout |
| `retry_attempts()` | `pub fn retry_attempts(mut self, attempts: u32) -> Self` | 🔍 | | Set retry count |

---

## 2. AUTHENTICATION METHODS

**ORIGINAL LOCATION**: `tmp/http3_public_api/packages/http3/src/builder/auth.rs`

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `api_key()` | `pub fn api_key(self, key: &str) -> Self` | ✅ ORIGINAL | `tmp/.../auth.rs:25` | Sets X-API-Key header |
| `basic_auth()` | `pub fn basic_auth(self, auth_config: impl Into<HashMap<&'static str, &'static str>>) -> Self` | ✅ ORIGINAL | `tmp/.../auth.rs:51` | **Accepts `[("user", "pass")]` syntax** |
| `bearer_auth()` | `pub fn bearer_auth(self, token: &str) -> Self` | ✅ ORIGINAL | `tmp/.../auth.rs:90` | Sets Bearer token |

---

## 3. HEADER METHODS  

**ORIGINAL LOCATION**: `tmp/http3_public_api/packages/http3/src/builder/headers.rs`

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `header()` | `pub fn header(mut self, key: HeaderName, value: HeaderValue) -> Self` | 🔍 | | Single header |
| `headers()` | `pub fn headers(mut self, headers_config: impl Into<HashMap<&'static str, &'static str>>) -> Self` | ✅ ORIGINAL | `tmp/.../headers.rs:103` | **CRITICAL: Accepts `[("key", "value")]` syntax** |
| `cache_control()` | `pub fn cache_control(self, value: &str) -> Self` | 🔍 | | Cache control header |
| `max_age()` | `pub fn max_age(self, seconds: u64) -> Self` | 🔍 | | Max age directive |
| `user_agent()` | `pub fn user_agent(self, user_agent: &str) -> Self` | 🔍 | | User agent header |
| `accept()` | `pub fn accept<T: Into<AcceptValue>>(self, accept: T) -> Self` | 🔍 | | Accept header |
| `accept_content_type()` | `pub fn accept_content_type(self, content_type: ContentType) -> Self` | 🔍 | | Accept content type |

---

## 4. BODY METHODS

**ORIGINAL LOCATION**: `tmp/http3_public_api/packages/http3/src/builder/body.rs`

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `body()` | `pub fn body<T: Serialize>(self, body: &T) -> Http3Builder<BodySet>` | 🔍 | | JSON serialized body |
| `raw_body()` | `pub fn raw_body(self, bytes: Vec<u8>) -> Http3Builder<BodySet>` | 🔍 | | Raw bytes body |
| `text_body()` | `pub fn text_body(self, text: &str) -> Http3Builder<BodySet>` | 🔍 | | Plain text body |

---

## 5. HTTP METHOD IMPLEMENTATIONS

**ORIGINAL LOCATION**: `tmp/http3_public_api/packages/http3/src/builder/methods.rs`

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `get()` | `pub fn get(mut self, url: &str) -> HttpStream` | 🔍 | | GET request |
| `post()` | `pub fn post(mut self, url: &str) -> HttpStream` | 🔍 | | POST request |
| `put()` | `pub fn put(mut self, url: &str) -> HttpStream` | 🔍 | | PUT request |
| `patch()` | `pub fn patch(mut self, url: &str) -> HttpStream` | 🔍 | | PATCH request |
| `delete()` | `pub fn delete(mut self, url: &str) -> HttpStream` | 🔍 | | DELETE request |
| `download_file()` | `pub fn download_file(mut self, url: &str) -> DownloadBuilder` | 🔍 | | File download |

### JSONPath HTTP Methods

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `get<T>()` | `pub fn get<T>(mut self, url: &str) -> JsonPathStream<T>` | 🔍 | | JSONPath GET |
| `post<T>()` | `pub fn post<T>(mut self, url: &str) -> JsonPathStream<T>` | 🔍 | | JSONPath POST |

---

## 6. STREAM COLLECTION METHODS (AsyncStream)

**STATUS**: Recently implemented in fluent-ai-async

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `collect_one()` | `pub fn collect_one(self) -> T` | ✅ IMPLEMENTED | `fluent-ai-async/.../receiver.rs` | Returns first item |
| `collect_one_or_else()` | `pub fn collect_one_or_else<F>(self, error_handler: F) -> T` | ✅ IMPLEMENTED | `fluent-ai-async/.../receiver.rs` | With error handling |

---

## 7. ERROR CHUNK CREATION

| Method | Original Signature | Status | Current Location | Notes |
|--------|-------------------|---------|------------------|--------|
| `HttpChunk::bad_chunk()` | `pub fn bad_chunk(error: String) -> Self` | 🔍 | | **CRITICAL**: Used in examples |

---

## CURRENT SEARCH STATUS

- ✅ **ORIGINAL**: Method exists in original revision a97141b  
- ✅ **IMPLEMENTED**: Method exists and works in current codebase
- 🔍 **SEARCHING**: Need to verify if method exists in current codebase
- ❌ **MISSING**: Method confirmed missing from current codebase
- ⚠️ **BROKEN**: Method exists but signature/behavior changed

---

## NEXT STEPS

1. 🔍 **SYSTEMATIC SEARCH**: Check EVERY method marked as "SEARCHING" in current codebase
2. 📊 **STATUS MATRIX**: Complete the EXISTS/MISSING/BROKEN analysis  
3. 🚨 **CRITICAL GAPS**: Identify must-have methods that are missing
4. 🔧 **RESTORATION PLAN**: Prioritize restoration based on fluent_builder example needs

**NO CODING WORK UNTIL THIS ANALYSIS IS 100% COMPLETE**

---

*Document Status: IN PROGRESS - Need to complete systematic search of current codebase*