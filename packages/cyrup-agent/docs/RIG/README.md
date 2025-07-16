# RIG Framework Domain Objects Directory

**Framework Version:** 0.13.0  
**Repository:** https://github.com/0xPlaygrounds/rig  
**Last Updated:** 2024

This directory contains documentation for individual domain objects in the RIG framework. Each subdirectory documents a single domain object with its properties and cross-references to related objects.

## Domain Objects

### Core Objects
- **[01-agent/](./01-agent/)** - Agent
- **[prompt-request/](./prompt-request/)** - PromptRequest
- **[tool-set/](./tool-set/)** - ToolSet
- **[message/](./message/)** - Message
- **[user-content/](./user-content/)** - UserContent
- **[assistant-content/](./assistant-content/)** - AssistantContent
- **[text/](./text/)** - Text
- **[image/](./image/)** - Image
- **[audio/](./audio/)** - Audio
- **[document/](./document/)** - Document
- **[tool-result/](./tool-result/)** - ToolResult
- **[tool-call/](./tool-call/)** - ToolCall
- **[tool-function/](./tool-function/)** - ToolFunction
- **[tool-definition/](./tool-definition/)** - ToolDefinition
- **[tool-type/](./tool-type/)** - ToolType
- **[mcp-tool/](./mcp-tool/)** - McpTool

### Model Objects
- **[completion-model/](./completion-model/)** - CompletionModel
- **[completion-request/](./completion-request/)** - CompletionRequest
- **[completion-response/](./completion-response/)** - CompletionResponse
- **[embedding-model/](./embedding-model/)** - EmbeddingModel
- **[embedding/](./embedding/)** - Embedding
- **[audio-generation-model/](./audio-generation-model/)** - AudioGenerationModel
- **[image-generation-model/](./image-generation-model/)** - ImageGenerationModel
- **[transcription-model/](./transcription-model/)** - TranscriptionModel

### Vector Store Objects
- **[vector-store-index/](./vector-store-index/)** - VectorStoreIndex
- **[vector-store-index-dyn/](./vector-store-index-dyn/)** - VectorStoreIndexDyn
- **[in-memory-vector-store/](./in-memory-vector-store/)** - InMemoryVectorStore
- **[in-memory-vector-index/](./in-memory-vector-index/)** - InMemoryVectorIndex

### Pipeline Objects
- **[pipeline-builder/](./pipeline-builder/)** - PipelineBuilder
- **[lookup-operation/](./lookup-operation/)** - Lookup
- **[prompt-operation/](./prompt-operation/)** - Prompt
- **[extract-operation/](./extract-operation/)** - Extract
- **[parallel-operation/](./parallel-operation/)** - Parallel
- **[map-operation/](./map-operation/)** - Map
- **[then-operation/](./then-operation/)** - Then
- **[sequential-operation/](./sequential-operation/)** - Sequential

### Builder Objects
- **[agent-builder/](./agent-builder/)** - AgentBuilder
- **[embeddings-builder/](./embeddings-builder/)** - EmbeddingsBuilder
- **[completion-request-builder/](./completion-request-builder/)** - CompletionRequestBuilder
- **[tool-set-builder/](./tool-set-builder/)** - ToolSetBuilder
- **[extractor-builder/](./extractor-builder/)** - ExtractorBuilder

### Provider Objects
- **[provider-client/](./provider-client/)** - ProviderClient
- **[openai-client/](./openai-client/)** - OpenAI Client
- **[anthropic-client/](./anthropic-client/)** - Anthropic Client
- **[cohere-client/](./cohere-client/)** - Cohere Client
- **[gemini-client/](./gemini-client/)** - Gemini Client

### Streaming Objects
- **[streaming-completion-response/](./streaming-completion-response/)** - StreamingCompletionResponse
- **[raw-streaming-choice/](./raw-streaming-choice/)** - RawStreamingChoice

### File Loader Objects
- **[file-loader/](./file-loader/)** - FileLoader
- **[pdf-file-loader/](./pdf-file-loader/)** - PdfFileLoader
- **[epub-file-loader/](./epub-file-loader/)** - EpubFileLoader

### Extractor Objects
- **[extractor/](./extractor/)** - Extractor

### Utility Objects
- **[one-or-many/](./one-or-many/)** - OneOrMany

### Error Objects
- **[completion-error/](./completion-error/)** - CompletionError
- **[prompt-error/](./prompt-error/)** - PromptError
- **[embedding-error/](./embedding-error/)** - EmbeddingError
- **[vector-store-error/](./vector-store-error/)** - VectorStoreError
- **[tool-error/](./tool-error/)** - ToolError
- **[tool-set-error/](./tool-set-error/)** - ToolSetError
- **[extraction-error/](./extraction-error/)** - ExtractionError
- **[file-loader-error/](./file-loader-error/)** - FileLoaderError
- **[client-build-error/](./client-build-error/)** - ClientBuildError
- **[message-error/](./message-error/)** - MessageError

## Documentation Format

Each domain object directory contains:
- **README.md** - Object properties table with cross-links and examples
- **BUILDER.md** - Builder patterns (if applicable)
- **RELATIONSHIPS.md** - Visual relationship diagrams (if applicable)

## Cross-References

Objects link to related objects using relative paths. For example:
- `[ToolSet](../tool-set/)` links to the ToolSet object
- `[Message](../message/)` links to the Message object

This creates a navigable web of domain object relationships.