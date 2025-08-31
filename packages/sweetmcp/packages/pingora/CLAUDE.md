# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SweetMCP Server (sweetmcp-pingora) is a production-grade, multi-protocol edge proxy built on Pingora 0.5. It normalizes GraphQL, JSON-RPC 2.0, and Cap'n Proto requests into Model Context Protocol (MCP) format, with intelligent load balancing across distributed peers.

## Core Architecture

### Multi-Protocol Gateway
- **Protocol Conversion**: Converts GraphQL queries, JSON-RPC 2.0 calls, and Cap'n Proto binary messages to MCP format
- **Edge Proxy**: Built on Cloudflare's Pingora 0.5 for high-performance request handling
- **Load Balancing**: Routes requests to local or peer nodes based on `node_load1` metrics

### Service Discovery & Distribution
- **DNS-based Discovery**: Primary discovery via SRV records with DoH (DNS over HTTPS)
- **mDNS Fallback**: Zero-config local network discovery
- **Peer Exchange**: HTTP-based mesh formation with build compatibility verification
- **Build ID Verification**: Ensures only identical builds form clusters using git hash + architecture

### Key Components
- `src/edge/`: Decomposed edge service with auth, core, and routing modules
- `src/peer_discovery.rs`: Distributed peer management and selection
- `src/mcp_bridge.rs`: Protocol normalization to MCP
- `src/auth.rs`: JWT-based authentication with HS256
- `src/metrics.rs`: Prometheus metrics collection
- `src/tls/`: Certificate management and OCSP validation

## Development Commands

### Building
```bash
cargo build --release                    # Production build
cargo build                            # Development build
```

### Testing
```bash
cargo test                             # Run all tests
cargo test --test unit                 # Run unit tests only
cargo test tls_tests                   # Run specific test file
```

### Quality Checks
```bash
cargo check                           # Fast compilation check
cargo clippy                          # Linting
cargo fmt                             # Code formatting
```

### Running
```bash
# Set required environment variable first
export SWEETMCP_JWT_SECRET=$(openssl rand -base64 32)
cargo run --release                   # Run the server
```

## Test Organization

**CRITICAL**: Tests must be placed in `tests/` directory, NOT in `src/` files:
- Unit tests: `tests/unit/<module_name>.rs`
- Integration tests: `tests/integration/<feature_name>.rs`
- Remove any `#[cfg(test)]` blocks from source files

## Configuration

### Required Environment
```bash
export SWEETMCP_JWT_SECRET=$(openssl rand -base64 32)
```

### Optional Configuration
```bash
export SWEETMCP_INFLIGHT_MAX=400
export SWEETMCP_TCP_BIND="0.0.0.0:8443"
export SWEETMCP_UDS_PATH="/run/sugora.sock"
export SWEETMCP_METRICS_BIND="127.0.0.1:9090"
export SWEETMCP_DNS_SERVICE="_sweetmcp._tcp.example.com"
export SWEETMCP_DOMAIN="example.com"
export SWEETMCP_DISCOVERY_TOKEN="shared-secret"
```

## Current Development Focus

### Panic Elimination Project
The codebase is currently undergoing comprehensive panic elimination with **101 identified unwrap()/expect() calls** that need replacement with proper error handling. See `TODO.md` for detailed tracking.

Key patterns being replaced:
- `mutex.lock().unwrap()` → proper poisoned lock recovery
- `Option::unwrap()` → explicit match or ok_or_else
- `Result::unwrap()` → ? operator or match with context
- Time arithmetic → saturating operations
- Certificate parsing → error propagation

### Quality Requirements
- Zero unwrap()/expect() calls in production code
- All errors must be handled gracefully with recovery or propagation
- Error messages must provide debugging context
- No functionality loss during error handling refactoring

## Architecture Patterns

### Service Decomposition
The `src/edge/` module is decomposed into sub-modules to maintain the 300-line limit:
- `auth/`: Authentication handling and JWT operations
- `core/`: Core EdgeService implementation
- `routing/`: Request routing and load balancing logic

### Background Services
Uses Pingora's background service pattern for:
- MCP bridge message processing
- DNS/mDNS discovery
- Peer exchange and health checking
- Rate limit cleanup
- Metrics collection
- TLS certificate management

### Build System
- Custom `build.rs` generates BUILD_ID from git hash + architecture
- Detects dirty working directory state
- Used for peer compatibility verification in distributed deployments

## Key Dependencies

- **pingora**: Core proxy framework from Cloudflare
- **tokio**: Async runtime (v1.47+)
- **sweetmcp-axum**: MCP protocol handling
- **opentelemetry**: Observability and metrics
- **jsonwebtoken**: JWT authentication
- **hickory-resolver**: DNS resolution with DoH support
- **ring**: Cryptographic operations

## Production Considerations

### Security
- Always set `SWEETMCP_DISCOVERY_TOKEN` in production
- Built-in rate limiting (10 req/min per endpoint)
- Automatic TCP health checks every 10s
- TLS certificates required for production deployment

### Monitoring
- Prometheus metrics available at `/metrics` endpoint
- OpenTelemetry tracing integration
- Load-based peer selection metrics
- Circuit breaker and rate limiting observability

### Performance
- Lock-free operations using `arc-swap` and `atomic-counter`
- Concurrent metrics collection from multiple peers
- Saturating time arithmetic for stability
- Connection pooling and keep-alive