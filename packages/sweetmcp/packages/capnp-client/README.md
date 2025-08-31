# SweetMCP Cap'n Proto Client

This package demonstrates the **real-world integration** of the SweetMCP protocol extension that supports GraphQL and Cap'n Proto protocols alongside the standard JSON-RPC MCP protocol.

## What This Demonstrates

🎯 **Complete Protocol Extension**: Shows how SweetMCP Pingora extends the Model Context Protocol (MCP) to support:

- **Cap'n Proto binary protocol** → JSON-RPC → MCP Tools → JSON-RPC → Cap'n Proto binary
- **GraphQL queries** → JSON-RPC → MCP Tools → JSON-RPC → GraphQL responses
- **Standard JSON-RPC** → MCP Tools (unchanged)

🔧 **Real MCP Plugin Integration**: Uses actual MCP plugins from the SweetMCP ecosystem:

- **Time Plugin**: Get current UTC time, parse time strings
- **Hash Plugin**: Generate SHA256, MD5, base64, base32 hashes

🚀 **Production-Quality Implementation**: All stub functions have been replaced with full implementations:

- ✅ Cap'n Proto binary detection with proper segment table validation
- ✅ GraphQL type condition validation with fragment compatibility checking  
- ✅ GraphQL response shaping with field mapping, aliases, and nested selections

## Architecture Overview

```
Cap'n Proto Client ──────┐
                         │
GraphQL Client ──────────┼───► SweetMCP Pingora Server
                         │           │
JSON-RPC Client ─────────┘           │
                                     ▼
                             Protocol Detection
                                     │
                                     ▼
                             JSON-RPC Normalization
                                     │
                                     ▼
                               MCP Tool Execution
                                  (time, hash)
                                     │
                                     ▼
                             Response Conversion
                                     │
                                     ▼
                          ┌─── Cap'n Proto Binary
                          ├─── GraphQL Response  
                          └─── JSON-RPC Response
```

## Real-World Examples

### Cap'n Proto Integration

The Cap'n Proto client shows the complete binary protocol integration:

```bash
# Run the Cap'n Proto demo
cargo run --bin sweetmcp-capnp-client

# This will:
# 1. Create Cap'n Proto binary requests for time and hash tools
# 2. Send them to SweetMCP Pingora server
# 3. Demonstrate protocol detection and conversion
# 4. Show real MCP plugin execution
# 5. Return Cap'n Proto binary responses
```

**Sample Output:**
```
🚀 Starting SweetMCP Cap'n Proto Integration Demo
📝 Demo 1: Testing Time Tool via Cap'n Proto
  📦 Created Cap'n Proto binary message (156 bytes)
  🌐 Sending request to SweetMCP server...
  📬 Received Cap'n Proto response:
    Request ID: 550e8400-e29b-41d4-a716-446655440000
    Status: success
    Time Data: {"utc_time":"1640995200","utc_time_rfc2822":"Sat, 01 Jan 2022 00:00:00 +0000"}
  ✅ Time tool returned valid timestamp data
```

### GraphQL Integration

The GraphQL demo shows query parsing, fragment resolution, and response shaping:

```bash
# Run the GraphQL demo
cargo run --example graphql_demo

# This will:
# 1. Send GraphQL queries with fragments and variables
# 2. Demonstrate GraphQL → JSON-RPC conversion
# 3. Show fragment resolution and type validation
# 4. Display properly shaped GraphQL responses
```

**Sample GraphQL Query:**
```graphql
fragment ToolResult on OperationResult {
    success
    timestamp
    execution_time
}

query MultipleOperations {
    timeOp: timeOperation(name: "get_time_utc") {
        ...ToolResult
        utc_time
        utc_time_rfc2822
    }
    
    hashOp: hashOperation(data: "Fragment Test", algorithm: "md5") {
        ...ToolResult
        hash_result
        algorithm_used
    }
}
```

## Cap'n Proto Schema

The client uses a comprehensive Cap'n Proto schema that maps to MCP tool concepts:

```capnp
struct McpToolRequest {
  requestId @0 :Text;
  toolName @1 :Text;
  arguments @2 :List(Argument);
  metadata @3 :Metadata;
  
  struct Argument {
    key @0 :Text;
    value @1 :ArgumentValue;
  }
  
  union ArgumentValue {
    text @0 :Text;
    number @1 :Float64;
    boolean @2 :Bool;
    listValue @3 :List(Text);
  }
}
```

## Technical Implementation Details

### Protocol Detection

The SweetMCP server automatically detects incoming protocol format:

1. **Content-Type Headers**: `application/capnp`, `application/graphql`
2. **Binary Format Analysis**: Cap'n Proto segment table validation
3. **JSON Structure**: GraphQL query detection vs JSON-RPC method calls

### Cap'n Proto Binary Validation

Proper Cap'n Proto detection based on the official specification:

- Segment count validation (0 < count < 512)
- Segment length validation with overflow protection
- Message size validation against segment table requirements
- Support for both packed and unpacked formats

### GraphQL Processing

Complete GraphQL support with:

- **Fragment Resolution**: Inline fragments and named fragment spreads
- **Type Validation**: GraphQL naming conventions and reserved type checking
- **Response Shaping**: Field selection, aliases, nested object handling
- **Variable Support**: Proper variable substitution and type checking

## Running the Examples

### Prerequisites

1. **SweetMCP Pingora Server** running on `localhost:8443`
2. **MCP Plugins** available (time, hash)
3. **Rust 2021 Edition** with required dependencies

### Build and Run

```bash
# Build the Cap'n Proto client
cd /path/to/sweetmcp/packages/capnp-client
cargo build

# Run Cap'n Proto integration demo
cargo run

# Run GraphQL integration demo  
cargo run --example graphql_demo

# Build for production
cargo build --release
```

### Expected Server Setup

The examples expect SweetMCP Pingora server to be running with:

- HTTP endpoint on port 8443
- MCP plugin support enabled
- Time and hash plugins loaded
- Protocol extension enabled

## Integration Verification

The examples include comprehensive verification that:

✅ **Cap'n Proto Binary Detection**: Proper segment table parsing  
✅ **GraphQL Type Validation**: Fragment and type condition checking  
✅ **Response Shaping**: Field mapping and alias handling  
✅ **MCP Plugin Integration**: Real tool execution with time and hash plugins  
✅ **Round-Trip Data Integrity**: Request → conversion → execution → conversion → response  
✅ **Error Handling**: Graceful handling of malformed inputs  

## Production Considerations

This implementation is **production-ready** with:

- **Zero Stubs**: All placeholder implementations replaced with full functionality
- **Comprehensive Error Handling**: No unwrap()/expect() calls in production paths  
- **Memory Safety**: Proper bounds checking and overflow protection
- **Security**: Input validation and sanitization
- **Performance**: Efficient binary parsing and response generation

## Contributing

When extending this client:

1. **Add New Protocols**: Follow the same pattern as Cap'n Proto and GraphQL
2. **Add MCP Plugins**: Create schema mappings for new tool types
3. **Enhance Validation**: Extend type checking and format validation
4. **Add Tests**: Include round-trip integration tests

This demonstrates the **real power** of the SweetMCP protocol extension - unified access to MCP tools through multiple protocol interfaces while maintaining full compatibility and production-quality implementation.