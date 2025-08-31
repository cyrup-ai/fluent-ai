@0x9eb32e19f86ee175;

# MCP Request Schema for Cap'n Proto client
# This schema defines the structure for making MCP tool requests via Cap'n Proto

struct McpToolRequest {
  # Unique request identifier
  requestId @0 :Text;
  
  # Name of the MCP tool to invoke
  toolName @1 :Text;
  
  # Tool arguments as key-value pairs
  arguments @2 :List(Argument);
  
  # Optional metadata
  metadata @3 :Metadata;
  
  struct Argument {
    key @0 :Text;
    value @1 :ArgumentValue;
    
    struct ArgumentValue {
      union {
        text @0 :Text;
        number @1 :Float64;
        boolean @2 :Bool;
        listValue @3 :List(Text);
      }
    }
  }
  
  struct Metadata {
    # Client information
    clientName @0 :Text;
    clientVersion @1 :Text;
    
    # Request context
    timestamp @2 :UInt64;
    
    # Protocol version
    protocolVersion @3 :Text;
  }
}

struct McpToolResponse {
  # Request ID this response corresponds to
  requestId @0 :Text;
  
  # Response status
  status @1 :ResponseStatus;
  
  # Tool execution result
  result @2 :ToolResult;
  
  # Any error information
  error @3 :ErrorInfo;
  
  enum ResponseStatus {
    success @0;
    error @1;
    timeout @2;
  }
  
  struct ToolResult {
    # Content returned by the tool
    content @0 :List(ContentItem);
    
    # Whether this result represents an error
    isError @1 :Bool;
  }
  
  struct ContentItem {
    # Content type (text, image, audio, resource)
    contentType @0 :Text;
    
    # Actual content data
    data @1 :Text;
    
    # Optional MIME type for binary content
    mimeType @2 :Text;
  }
  
  struct ErrorInfo {
    code @0 :Int32;
    message @1 :Text;
    details @2 :Text;
  }
}