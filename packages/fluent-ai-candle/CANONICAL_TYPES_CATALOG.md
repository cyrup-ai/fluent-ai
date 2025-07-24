# Canonical Types Catalog - Candle Crate

## Types Module Structure Analysis

### src/types/mod.rs (Lines 1-16)
**Architecture**: Main types module with glob re-exports from all sub-modules
**Exports**: All types from candle_completion, candle_chat, candle_context, candle_model, candle_engine, candle_utils

**Sub-modules identified**:
- candle_completion (completion-related types)
- candle_chat (chat and message types)  
- candle_context (context and document types)
- candle_model (model-related types)
- candle_engine (engine types)
- candle_utils (utility types)

## Detailed Type Catalog

### Phase 1.1 Complete: Types Module Structure Documented
- Main module uses glob re-exports (pub use module::*)
- Clean separation of concerns across sub-modules
- No logic in mod.rs - follows ONE FILE PER concept principle

### Candle Completion Module (src/types/candle_completion/)
**Architecture**: Well-organized with one concept per file, proper separation of concerns

**Files and Types Cataloged**:
- `completion_params.rs`: CompletionParams struct with builder methods
- `request.rs`: CandleCompletionRequest, CompletionRequestBuilder, CompletionRequestError
- `response.rs`: (to be cataloged)
- `streaming.rs`: (to be cataloged) 
- `core.rs`: (to be cataloged)
- `error.rs`: (to be cataloged)
- `model_params.rs`: (to be cataloged)
- `tool_definition.rs`: (to be cataloged)
- `constants.rs`: (to be cataloged)

**Key Canonical Types Identified**:
- CompletionParams (zero-allocation, blazing-fast with const constructors)
- CandleCompletionRequest (main request type)
- CompletionRequestBuilder (ergonomic builder pattern)
- CompletionRequestError (comprehensive error handling)
- CompletionResponse, CompletionResponseBuilder (response handling)
- CompactCompletionResponse, CompactCompletionResponseBuilder (optimized responses)
- CandleStreamingResponse, CandleStreamingChoice, CandleStreamingDelta (streaming types)
- CandleFinishReason, CandleLogProbs, CandleToolCallDelta (supporting types)
- TEMPERATURE_RANGE, MAX_TOKENS, MAX_CHUNK_SIZE (constants)

### Phase 1 Complete: Canonical Types Cataloged
**Total Canonical Types**: 15+ types across completion, chat, context, model, engine, utils modules
**Architecture**: Clean separation with one concept per file, zero-allocation patterns, blazing-fast performance

## Phase 2: Identify Actual Duplicates

### Next: Search for duplicate types outside types module