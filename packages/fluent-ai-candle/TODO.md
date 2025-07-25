# TODO: Critical Compilation Fixes Required

## CURRENT STATUS: 🚨 COMPILATION BROKEN  
**749 ERRORS** - Project cannot compile successfully. Core module structure needs systematic repair.

## QA FINDINGS FROM COMPREHENSIVE REVIEW

### ✅ VERIFIED COMPLETED WORK (PASSED QA)
- **Core Streaming Functions**: tokenizer, streaming/flow_control, model/loading - all AsyncStream compliant
- **Module Decomposition**: parsing, processing/error, error modules - successfully decomposed into focused modules ≤300 lines
- **AsyncStream Architecture**: Most async/await patterns successfully eliminated

### 🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION



#### 6. Address 45 Additional Compilation Errors
**Focus Areas**:
- Unresolved imports throughout the codebase
- Type mismatches and lifetime issues  
- Missing struct fields and method signatures
- Module visibility and re-export issues

**Reference**: Use `cargo check` output for complete error list and specific line numbers

#### 7. Clean Up 125 Compilation Warnings
**Priority**: Address after critical errors are fixed
**Focus**: Remove unused imports, unused variables, deprecated type aliases

## IMPLEMENTATION APPROACH

1. **IMMEDIATE (30 min)**: Fix syntax errors preventing compilation
2. **HIGH PRIORITY (1 hour)**: Fix missing imports and module issues  
3. **MEDIUM PRIORITY (1 hour)**: Resolve type conflicts and trait implementations
4. **FINAL CLEANUP (30 min)**: Address remaining errors and critical warnings

## SUCCESS CRITERIA

- ✅ `cargo check` completes with 0 errors
- ✅ AsyncStream architecture maintained (no Box::pin async patterns)
- ✅ All modules compile cleanly
- ✅ Warnings reduced to <10 non-critical items

---

## ASYNC/AWAIT VIOLATIONS - STREAMS ONLY ARCHITECTURE

**CRITICAL**: Every async/await violation MUST be converted to AsyncStream patterns with the following requirements:

### 🚨 MANDATORY ARCHITECTURE RULES - BASED ON fluent-ai-async:
1. **STREAMS ONLY** - NO Future types, NO async/await anywhere
2. **ALL VALUES UNWRAPPED** - Process with .on_chunk() handlers, never Result<T,E>  
3. **USE emit!/handle_error! MACROS ONLY** - No Result returns from closures
4. **NO async fn signatures** - Return AsyncStream<T> from regular fn
5. **USE .collect() or .on_chunk()** - Replace all .await with these patterns
6. **AsyncStream::with_channel()** - Primary constructor for all streams
7. **AsyncTask::collect()** - Replace .await for single values

### ✅ CORRECT PATTERNS FROM fluent-ai-async:
```rust
// CORRECT - AsyncStream with unwrapped values and emit! macro
pub fn operation() -> AsyncStream<ValueType> {
    use fluent_ai_async::{AsyncStream, emit, handle_error};
    
    AsyncStream::with_channel(move |sender| {
        loop {
            match do_work() {
                Ok(value) => emit!(sender, value),  // emit! sends unwrapped value
                Err(e) => handle_error!(e, "work failed"), // Terminates stream gracefully
            }
        }
        // NO return statement - macros handle everything
    })
}

// CORRECT - Consumption with on_chunk for unwrapped values
stream.on_chunk(|unwrapped_value| {
    // Process each value directly, NO Result unwrapping needed
    process(unwrapped_value);
});

// CORRECT - Collect all values when needed (replaces .await)
let all_values: Vec<ValueType> = stream.collect();

// CORRECT - Single value task (replaces async fn)
pub fn single_operation() -> AsyncTask<ResultType> {
    use fluent_ai_async::spawn_task;
    
    spawn_task(|| {
        // Synchronous work only
        do_work()  // Return unwrapped value directly
    })
}

// CORRECT - Consume single value (replaces .await)
let result = task.collect();  // Blocking collection
```

### ❌ FORBIDDEN PATTERNS:
```rust
// WRONG - async fn
pub async fn operation() -> Result<T, E> { ... }

// WRONG - Result inside stream  
AsyncStream<Result<T, E>>

// WRONG - returning any value from with_channel closure
AsyncStream::with_channel(move |sender| {
    Ok(())      // FORBIDDEN - NO RETURN VALUES
    // or
    return;     // FORBIDDEN - only emit!/handle_error! should return
})

// WRONG - .await usage anywhere
value.operation().await

// WRONG - Box::pin(async move {})
Box::pin(async move { ... })

// WRONG - tokio::spawn or async runtime usage
tokio::spawn(async { ... })

// WRONG - Manual Result handling in streams
if let Ok(value) = result {
    sender.send(Ok(value)).unwrap(); // NEVER wrap in Result
}
```

### 📋 VIOLATIONS REQUIRING IMMEDIATE FIX:

#### CRITICAL: src/types/candle_chat/macros/playbook.rs (9 violations)
**MANDATORY fluent-ai-async conversion - STREAMS ONLY, NO FUTURES:**

**File must be converted to return AsyncStream<PlaybackEvent>:**
```rust
pub fn execute_playback() -> AsyncStream<PlaybackEvent> {
    use fluent_ai_async::{AsyncStream, emit, handle_error};
    
    AsyncStream::with_channel(move |sender| {
        // All work is synchronous inside closure
    })
}
```

**Specific Line Conversions:**
- **Line 66**: `self.playbook_sessions.write().await` → `self.playbook_sessions.write().expect("lock poisoned")`  
- **Line 77**: `self.playbook_sessions.write().await` → `self.playbook_sessions.write().expect("lock poisoned")`
- **Line 94**: `self.execute_single_action(...).await` → `self.execute_single_action(...).on_chunk(|result| emit!(sender, result))`
- **Line 108**: `tokio::time::sleep(duration).await` → `std::thread::sleep(duration)` 
- **Line 174**: `self.execute_single_action(...).await` → `self.execute_single_action(...).on_chunk(|result| emit!(sender, result))`
- **Line 201**: `self.execute_single_action(...).await` → `self.execute_single_action(...).on_chunk(|result| emit!(sender, result))`
- **Line 222**: `self.playbook_sessions.write().await` → `self.playbook_sessions.write().expect("lock poisoned")`
- **Line 241**: `self.playbook_sessions.write().await` → `self.playbook_sessions.write().expect("lock poisoned")`
- **Line 260**: `self.playbook_sessions.read().await` → `self.playbook_sessions.read().expect("lock poisoned")`

**Pattern**: Replace ALL async/await with synchronous operations inside AsyncStream::with_channel()

#### CRITICAL: src/types/candle_chat/macros/recording.rs (10 violations)  
**MANDATORY fluent-ai-async conversion - STREAMS ONLY, NO FUTURES:**

**File must be converted to return AsyncStream<RecordingEvent>:**
```rust
pub fn execute_recording() -> AsyncStream<RecordingEvent> {
    use fluent_ai_async::{AsyncStream, emit, handle_error};
    
    AsyncStream::with_channel(move |sender| {
        // All work is synchronous inside closure
        // Use emit!(sender, event) for each recording event
    })
}
```

**Specific Line Conversions:**
- **Line 89**: `self.recording_sessions.write().await` → `self.recording_sessions.write().expect("lock poisoned")`
- **Line 101**: `self.recording_sessions.read().await` → `self.recording_sessions.read().expect("lock poisoned")`
- **Line 119**: `self.recording_sessions.write().await` → `self.recording_sessions.write().expect("lock poisoned")`
- **Line 138**: `self.recording_sessions.write().await` → `self.recording_sessions.write().expect("lock poisoned")`
- **Line 157**: `self.recording_sessions.write().await` → `self.recording_sessions.write().expect("lock poisoned")`
- **Line 192**: `self.recording_sessions.read().await` → `self.recording_sessions.read().expect("lock poisoned")`
- **Line 203**: `self.recording_sessions.read().await` → `self.recording_sessions.read().expect("lock poisoned")`
- **Line 209**: `self.recording_sessions.write().await` → `self.recording_sessions.write().expect("lock poisoned")`
- **Line 220**: `self.recording_sessions.read().await` → `self.recording_sessions.read().expect("lock poisoned")`
- **Line 243**: `self.recording_sessions.write().await` → `self.recording_sessions.write().expect("lock poisoned")`

**Pattern**: Replace ALL tokio/async locks with std::sync synchronous locks inside AsyncStream::with_channel()


#### CRITICAL: src/types/candle_chat/chat/integrations/external.rs (6 violations)
**MANDATORY fluent-ai-async + fluent_ai_http3 conversion - STREAMS ONLY, NO FUTURES:**

**File must be converted to return AsyncStream<IntegrationEvent>:**
```rust
pub fn execute_integration() -> AsyncStream<IntegrationEvent> {
    use fluent_ai_async::{AsyncStream, emit, handle_error};
    use fluent_ai_http3::Http3;
    
    AsyncStream::with_channel(move |sender| {
        // Use Http3 streaming patterns - NO .await
        Http3::json()
            .body(&request)
            .post(url)
            .on_chunk(|chunk| {
                emit!(sender, IntegrationEvent::HttpResponse(chunk));
            });
    })
}
```

**Specific Line Conversions:**
- **Line 72**: `self.execute_http_request(request).await` → `self.execute_http_request(request).on_chunk(|response| emit!(sender, response))`
- **Line 74**: `self.execute_plugin_request(request).await` → `self.execute_plugin_request(request).on_chunk(|result| emit!(sender, result))`
- **Line 75**: `self.execute_service_request(request).await` → `self.execute_service_request(request).on_chunk(|result| emit!(sender, result))`
- **Line 148**: `.send().await` → Use `Http3::json().post().on_chunk()` pattern with fluent_ai_http3
- **Line 161**: `response.json().await` → Use `response.on_chunk()` with fluent_ai_http3 streaming
- **Line 244**: `self.execute_request(test_request).await` → `self.execute_request(test_request).on_chunk(|result| emit!(sender, result))`

**Pattern**: Replace ALL HTTP .await with fluent_ai_http3 streaming .on_chunk() patterns inside AsyncStream::with_channel()

#### CRITICAL: src/types/candle_chat/chat/integrations/manager.rs (4 violations)
**MANDATORY fluent-ai-async conversion - STREAMS ONLY, NO FUTURES:**

**ALL async fn methods must be converted to fn returning AsyncStream<T>:**
```rust
// WRONG - async fn
pub async fn manage_integration(&self) -> Result<IntegrationResult, Error> { ... }

// CORRECT - AsyncStream
pub fn manage_integration(&self) -> AsyncStream<IntegrationResult> {
    use fluent_ai_async::{AsyncStream, emit, handle_error};
    
    AsyncStream::with_channel(move |sender| {
        match self.process_integration() {
            Ok(result) => emit!(sender, result),  // Unwrapped value
            Err(e) => handle_error!(e, "integration failed"),
        }
    })
}
```

**Pattern**: Convert ALL async fn signatures to fn returning AsyncStream<T>, use emit! for values and handle_error! for failures

### 🎯 COMPLETION CRITERIA:
- ✅ Zero .await calls in production code  
- ✅ Zero async fn signatures in production code
- ✅ All operations return AsyncStream<T> (never AsyncStream<Result<T,E>>)
- ✅ All processing uses .on_chunk() handlers with unwrapped values

---

## REFERENCE FILES FOR CONTEXT

- **HTTP3 patterns**: `/../../tmp/candle/` (if available)
- **AsyncStream docs**: Use fluent-ai-async crate documentation  
- **Architecture guide**: `CLAUDE.md` for AsyncStream patterns and HTTP3 client usage