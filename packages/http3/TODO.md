# TODO: Http3 Fluent Builder Implementation Plan

## Milestone 1: Core Builder & Typestate Foundation

- **Task 1.1**: In `src/builder.rs`, define the core typestate marker traits (`MethodState`, `HeadersState`, `BodyState`, `UrlState`) and their corresponding state structs (e.g., `MethodNotSet`, `MethodSet`). This will form the compile-time safety backbone of the builder. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 1.2**: Act as an Objective QA Rust developer: Review the typestate definitions in `src/builder.rs`. Verify that the states logically represent the request building flow and prevent invalid state transitions. Confirm that the design is purely structural and contains no implementation logic. Rate the work performed previously on these requirements.

- **Task 1.3**: In `src/builder.rs`, define the primary `Http3Builder<M, H, B, U>` struct. It will be generic over the state markers and will contain an `HttpRequest` instance to be configured. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 1.4**: Act as an Objective QA Rust developer: Review the `Http3Builder` struct definition. Ensure it correctly uses `PhantomData` for the generic state markers and properly encapsulates the `HttpRequest`. Rate the work performed previously on these requirements.

- **Task 1.5**: In `src/builder.rs`, implement the `Http3` entry point struct with static methods `json()`, `form_urlencoded()`, and a generic `builder()`. These methods will instantiate `Http3Builder` in its initial state, pre-configuring headers as appropriate. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 1.6**: Act as an Objective QA Rust developer: Review the `Http3` entry point methods. Verify they correctly initialize the `Http3Builder` with the appropriate headers and initial typestate for `json()` and `form_urlencoded()`. Rate the work performed previously on these requirements.

## Milestone 2: Configuration Methods & `headers!` Macro

- **Task 2.1**: In `src/common/headers.rs`, define a comprehensive, type-safe `HeaderName` enum, mirroring the approach in `reqwest::header`. Include common headers like `ContentType`, `Accept`, `Authorization`, and custom ones like `XApiKey`. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 2.2**: Act as an Objective QA Rust developer: Review the `HeaderName` enum. Confirm it includes all necessary standard and custom headers and provides a type-safe way to reference them, preventing typos. Rate the work performed previously on these requirements.

- **Task 2.3**: In `src/builder.rs`, implement the `headers!{}` macro. This macro must accept the `HeaderName::Variant => "value"` syntax and expand to a `HashMap<String, String>`. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 2.4**: Act as an Objective QA Rust developer: Review the `headers!{}` macro implementation. Test its syntax and verify that it correctly produces a `HashMap` compatible with the builder's `.headers()` method. Rate the work performed previously on these requirements.

- **Task 2.5**: In `src/builder.rs`, implement the configuration methods on `Http3Builder`: `.headers(HashMap<String, String>)`, `.api_key(&str)`, `.basic_auth(&str, &str)`, and `.bearer_auth(&str)`. These methods will correctly modify the internal `HttpRequest` and transition the builder's state. The auth methods will leverage the existing providers in `src/common/auth.rs`. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 2.6**: Act as an Objective QA Rust developer: Review the configuration methods. Verify that headers are correctly applied and that authentication methods correctly format the `Authorization` header. Check that the typestate transitions are logical. Rate the work performed previously on these requirements.

- **Task 2.7**: In `src/builder.rs`, implement the generic `.body<T: Serialize>(&T)` method. This method will serialize the provided data to `Vec<u8>` (using `serde_json`) and store it in the `HttpRequest`. This method must return a `HttpResult<Http3Builder<...>>` as serialization can fail, and it will transition the builder to the `BodySet` state. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 2.8**: Act as an Objective QA Rust developer: Review the `.body()` method. Verify that it correctly handles serialization errors and that the typestate transition to `BodySet` is correctly implemented. Rate the work performed previously on these requirements.

## Milestone 3: Execution & Response Handling

- **Task 3.1**: In `src/builder.rs`, implement the terminal methods: `.get(url)`, `.post(url)`, `.put(url)`, etc. These methods will consume the builder, set the final URL and method on the `HttpRequest`, and use the `global_client()` to delegate execution to the appropriate module in `src/operations/`. The return type will be a new `ResponseStream` struct. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 3.2**: Act as an Objective QA Rust developer: Review the terminal methods. Verify that they consume the builder, correctly construct the final `HttpRequest`, and delegate to the `operations` modules. Check that the `ResponseStream` is returned. Rate the work performed previously on these requirements.

- **Task 3.3**: In `src/stream.rs`, define the `ResponseStream` struct. This struct will wrap the underlying `HttpStream`. Implement the `Stream` trait for it, yielding `HttpResult<HttpChunk>`. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 3.4**: Act as an Objective QA Rust developer: Review the `ResponseStream` struct and its `Stream` implementation. Ensure it correctly wraps the underlying stream and propagates chunks and errors. Rate the work performed previously on these requirements.

- **Task 3.5**: In `src/stream.rs`, implement the final consumer methods on `ResponseStream`: `.collect<T: DeserializeOwned>()` and `.collect_or_else<T: DeserializeOwned, F: FnOnce(HttpError) -> T>(f: F)`. The `collect` method will buffer the full response and deserialize it, returning `HttpResult<T>`. `collect_or_else` will allow for a fallback value or custom error handling on deserialization failure. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 3.6**: Act as an Objective QA Rust developer: Review the `.collect()` and `.collect_or_else()` methods. Verify that they correctly handle the asynchronous stream, buffer the response, and perform deserialization with robust error handling. Rate the work performed previously on these requirements.

## Milestone 4: Integration & Testing

- **Task 4.1**: Create a new test file `tests/builder_api.rs`. Add comprehensive integration tests that validate the entire fluent API, using each of the user's provided examples as a test case. Tests must use a real HTTP endpoint (e.g., httpbin.org) to verify requests and responses. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 4.2**: Act as an Objective QA Rust developer: Review the integration tests in `tests/builder_api.rs`. Verify that all user examples are covered and that the tests make real network requests to validate the end-to-end functionality of the builder. Ensure `expect()` is used for assertions as appropriate in a test context. Rate the work performed previously on these requirements.

- **Task 4.3**: In `src/lib.rs`, publicly export the main builder components: `Http3`, `Http3Builder`, and any necessary response types, ensuring they are discoverable and ergonomic for crate users. DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- **Task 4.4**: Act as an Objective QA Rust developer: Review the public exports in `src/lib.rs`. Confirm that the builder API is exposed cleanly and logically to end-users. Rate the work performed previously on these requirements.