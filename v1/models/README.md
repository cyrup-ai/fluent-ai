# API Response Structures Documentation

This directory contains actual API responses from each provider's `/v1/models` endpoint to document the structure differences.

## Provider Endpoints and Structures

### OpenAI (`openai.json`)
- **Endpoint**: `https://api.openai.com/v1/models`
- **Auth**: Bearer token required
- **Structure**: Standard OpenAI format
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4-0613",
      "object": "model", 
      "created": 1686588896,
      "owned_by": "openai"
    }
  ]
}
```

### Mistral (`mistral.json`)
- **Endpoint**: `https://api.mistral.ai/v1/models`
- **Auth**: Bearer token required
- **Structure**: Extended format with capabilities and pricing info
```json
{
  "object": "list",
  "data": [
    {
      "id": "mistral-medium-2505",
      "object": "model",
      "created": 1754042468,
      "owned_by": "mistralai",
      "capabilities": {
        "completion_chat": true,
        "completion_fim": false,
        "function_calling": true,
        "fine_tuning": false,
        "vision": true,
        "classification": false
      },
      "name": "mistral-medium-2505",
      "description": "Official mistral-medium-2505 Mistral AI model",
      "max_context_length": 131072,
      "aliases": ["mistral-medium-latest", "mistral-medium"],
      "default_model_temperature": 0.3,
      "type": "base"
    }
  ]
}
```

### Together.ai (`together.json`)
- **Endpoint**: `https://api.together.xyz/v1/models`
- **Auth**: Bearer token required
- **Structure**: Detailed format with pricing, configuration, and organization info
```json
[
  {
    "id": "mistralai/Mistral-7B-Instruct-v0.3",
    "object": "model",
    "created": 1716406261,
    "type": "chat",
    "running": false,
    "display_name": "Mistral (7B) Instruct v0.3",
    "organization": "mistralai",
    "link": "https://huggingface.co/api/models/mistralai/Mistral-7B-Instruct-v0.3",
    "license": "apache-2.0",
    "context_length": 32768,
    "config": {
      "chat_template": "...",
      "stop": ["</s>"],
      "bos_token": "<s>",
      "eos_token": "</s>"
    },
    "pricing": {
      "hourly": 0,
      "input": 0.2,
      "output": 0.2,
      "base": 0,
      "finetune": 0
    }
  }
]
```

### X.ai (`xai.json`)
- **Endpoint**: `https://api.x.ai/v1/models`
- **Auth**: Bearer token required
- **Structure**: Simple OpenAI-compatible format
```json
{
  "data": [
    {
      "id": "grok-2-1212",
      "created": 1737331200,
      "object": "model",
      "owned_by": "xai"
    }
  ],
  "object": "list"
}
```

### Huggingface (`huggingface.json`)
- **Endpoint**: `https://huggingface.co/api/models?filter=text-generation&sort=downloads&direction=-1&limit=50`
- **Auth**: None required (public API)
- **Structure**: Array of model metadata with download stats
```json
[
  {
    "_id": "621ffdc036468d709f17434d",
    "id": "openai-community/gpt2",
    "likes": 2865,
    "private": false,
    "downloads": 14347531,
    "tags": ["transformers", "pytorch", "text-generation", "en"],
    "pipeline_tag": "text-generation",
    "library_name": "transformers",
    "createdAt": "2022-03-02T23:29:04.000Z",
    "modelId": "openai-community/gpt2"
  }
]
```

## Key Differences

1. **OpenAI & X.ai**: Use standard `/v1/models` OpenAI format with minimal fields
2. **Mistral**: Extended format with capabilities, descriptions, aliases, and context lengths
3. **Together.ai**: Most detailed with pricing, configuration, licenses, and organization info
4. **Huggingface**: Completely different format focused on model metadata and download statistics
5. **Anthropic**: No API endpoint - uses hardcoded static models only

## Usage in Code Generation

The buildlib system must handle these different structures when converting to the unified `ModelData` format:
```rust
type ModelData = (String, u64, f64, f64, bool, Option<f64>);
// (id, context_length, input_price, output_price, is_thinking, thinking_price)
```