# TLS Integration into Production Daemon Pipeline

## Architecture Overview

**CORRECTED FINDINGS FROM CODE ANALYSIS:**

1. **Actual Architecture (from `/packages/daemon/src/installer/config.rs`):**
   - Line 123: "internal:pingora" configured as "Special command handled internally"
   - Line 248: service_type set to "proxy" for "sweetmcp-pingora"
   - Both autoconfig and pingora are designed to be handled internally by daemon

2. **Missing Implementation (from `/packages/daemon/src/service.rs`):**
   - Lines 140-142: Only autoconfig service_type handler exists
   - NO handler for service_type == "proxy" to spawn pingora services
   - Missing pingora.rs service module in `/packages/daemon/src/service/`

3. **Correct Pattern (from `/packages/daemon/src/service/autoconfig.rs`):**
   - Internal services run within daemon process using tokio runtime
   - Handle commands through channels and send events through bus
   - Use cancellation tokens for graceful shutdown

**SOLUTION:** Create missing pingora service module and add service_type == "proxy" handler to spawn() function, following autoconfig pattern.

## Implementation Tasks

### 1. Create Pingora Service Module
**File:** `/packages/daemon/src/service/pingora.rs` (new file)
**Architecture:** Internal service module similar to autoconfig.rs pattern
- [ ] Create PingoraService struct with name, bus, and service definition
- [ ] Implement tokio runtime for async operations within daemon process
- [ ] Use TLS builder API to generate certificates before starting pingora
- [ ] Spawn sweetmcp_server binary with TLS environment variables
- [ ] Handle Start/Stop/Restart/Shutdown commands through channels
- [ ] Send service state events through bus (running, stopped, fatal)
- [ ] Implement graceful shutdown with cancellation tokens

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify service follows autoconfig pattern, handles TLS generation, and spawns binary correctly.

### 2. Add Proxy Service Handler
**File:** `/packages/daemon/src/service.rs` lines 140-142
**Architecture:** Extend existing spawn() function to handle service_type == "proxy"
- [ ] Add condition for `def.service_type == Some("proxy".to_string())` alongside autoconfig check
- [ ] Import pingora module: `mod pingora;`
- [ ] Call `pingora::spawn_pingora(def, bus)` for proxy service types
- [ ] Maintain existing autoconfig handler unchanged

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify handler correctly identifies proxy service types and calls pingora spawn function.

### 3. Update Daemon Dependencies
**File:** `/packages/daemon/Cargo.toml`
**Architecture:** Add sweetmcp-pingora dependency for TLS builder access
- [ ] Add `sweetmcp-pingora = { path = "../pingora" }` to dependencies section
- [ ] Ensure no version conflicts with existing dependencies

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify dependency declaration is correct and path is accurate.

### 4. Implement TLS Certificate Generation
**File:** `/packages/daemon/src/service/pingora.rs` (continuation of task 1)
**Architecture:** Use TLS builder API to generate certificates before spawning binary
- [ ] Create TLS directory using environment variable or default path
- [ ] Load or create CA using `Tls::authority().path().load().await`
- [ ] Generate server certificate using `Tls::certificate().domain().authority().generate().await`
- [ ] Extract domain from service configuration or use default
- [ ] Handle TLS generation errors with proper error reporting

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify TLS generation uses real API calls, handles errors properly, and creates valid certificates.

### 5. Configure Binary Execution
**File:** `/packages/daemon/src/service/pingora.rs` (continuation of task 1)
**Architecture:** Spawn sweetmcp_server binary with TLS environment variables
- [ ] Determine binary path: check target/debug/sweetmcp_server or installed location
- [ ] Set TLS environment variables: SWEETMCP_TLS_DIR, SWEETMCP_SERVER_CERT, SWEETMCP_SERVER_KEY
- [ ] Use `Command::new()` with actual binary path and environment variables
- [ ] Set working directory from service definition
- [ ] Configure stdout/stderr handling for logging
- [ ] Handle binary spawn errors with proper error reporting

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify binary spawning uses correct path, sets environment variables, and handles errors properly.

### 6. Update Pingora Main for Environment Variables
**File:** `/packages/pingora/src/main.rs` lines 140-190 (existing TLS integration)
**Architecture:** Use environment variables from daemon instead of hardcoded paths
- [ ] Check for `SWEETMCP_TLS_DIR` environment variable first
- [ ] Use `SWEETMCP_SERVER_CERT` and `SWEETMCP_SERVER_KEY` if provided
- [ ] Fallback to existing TLS generation logic if environment variables not set
- [ ] Remove temporary file writing since daemon provides certificate paths
- [ ] Maintain backward compatibility for standalone operation

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify environment variable usage is correct and fallback logic works properly.

### 7. Test Integration
**File:** `/packages/daemon/tests/pingora_integration.rs` (new file)
**Architecture:** Integration test for pingora service module
- [ ] Create test that configures service_type == "proxy" service
- [ ] Verify TLS certificates are generated correctly
- [ ] Test that sweetmcp_server binary starts successfully
- [ ] Verify service responds to start/stop commands
- [ ] Test HTTPS functionality with generated certificates

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify integration test covers critical paths and validates TLS functionality.

## Success Criteria

1. **Start daemon with "internal:pingora" service and it successfully loads TLS certificates**
   - Daemon recognizes service_type == "proxy"
   - Pingora service module generates certificates successfully
   - sweetmcp_server binary starts without errors

2. **Pingora service serves HTTPS traffic with certificates from TLS builder**
   - HTTPS listeners are configured with generated certificates
   - Service responds to HTTPS requests
   - Certificate validation passes

## Architecture Notes

- **Internal Service Pattern:** Follow autoconfig.rs pattern for consistency
- **Binary Execution:** Spawn actual sweetmcp_server binary from within service module
- **TLS Integration:** Use existing TLS builder API without modifications
- **Environment Variables:** Clean interface between daemon and pingora for TLS config
- **Error Handling:** Proper error propagation through existing service infrastructure
- **Service Types:** autoconfig → "autoconfig", pingora → "proxy"