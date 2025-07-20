# CRITICAL: Domain Duplication Removal Tasks

## USER OBJECTIVE: Fix ALL warnings and errors in workspace (0 errors, 0 warnings)

### APPROVED PLAN: Systematic Domain Duplication Removal

**CRITICAL ARCHITECTURAL RULE**: ALL domain value types MUST be imported from `./packages/domain/**` in all packages. Absolutely NO duplication of domain types is allowed.

## Current Status
- Compilation errors reduced from 322 to 246 after USER fixes to agent/chat.rs and memory/manager.rs
- 24 files identified with duplicated domain models in provider clients
- Progress: Removed duplicated ModelInfo and CompletionModel from several clients

## Remaining Tasks (EXECUTE IMMEDIATELY)

### 1. Complete Domain Model Duplication Removal
**Files with duplicated domain models (24 total):**
- packages/provider/src/clients/azure/embedding.rs - EmbeddingModel struct
- packages/provider/src/clients/azure/image_generation.rs - ImageGenerationModel struct  
- packages/provider/src/clients/azure/transcription.rs - TranscriptionModel struct
- packages/provider/src/clients/azure/audio_generation.rs - AudioGenerationModel struct
- packages/provider/src/clients/candle/client.rs - Model structs
- packages/provider/src/clients/candle/config.rs - Model structs
- packages/provider/src/clients/candle/models.rs - Model structs
- packages/provider/src/clients/candle/model_cache.rs - Model structs
- packages/provider/src/clients/candle/model_repo.rs - Model structs
- packages/provider/src/clients/candle/kv_cache.rs - Model structs
- packages/provider/src/clients/vertexai/config.rs - Model struct
- packages/provider/src/clients/vertexai/models.rs - Model struct
- packages/provider/src/clients/huggingface/image_generation.rs - ImageGenerationModel struct
- packages/provider/src/clients/huggingface/transcription.rs - TranscriptionModel struct
- packages/provider/src/clients/gemini/embedding.rs - EmbeddingModel struct (partially fixed)
- packages/provider/src/clients/gemini/completion_old.rs - CompletionModel struct
- packages/provider/src/clients/gemini/transcription.rs - TranscriptionModel struct
- packages/provider/src/clients/openrouter/completion.rs - CompletionModel struct
- packages/provider/src/clients/together/embedding.rs - EmbeddingModel struct (partially fixed)
- packages/provider/src/clients/mistral/embedding.rs - EmbeddingModel struct (partially fixed)

### 2. Add Canonical Domain Imports
**Action**: Add imports from fluent_ai_domain::model::* to all provider client files
- ModelInfo from fluent_ai_domain::model::ModelInfo
- ModelCapabilities from fluent_ai_domain::model::ModelCapabilities
- ModelRegistry from fluent_ai_domain::model::ModelRegistry
- ModelError, Result from fluent_ai_domain::model::{ModelError, Result}

### 3. Fix Compilation Errors from Type Mismatches
**Current error patterns (246 total):**
- Provider build script issues (missing functions in string_utils, client_verifier)
- Domain crate type mismatches (MemoryConfig conflicts, missing trait implementations)
- Agent chat issues (missing store_memory method, type mismatches)
- Chat command issues (Result type mismatches, moved values)

### 4. Ensure Zero Domain Duplication (QA Pass)
**Verification steps:**
- Run: `find packages -name "*.rs" -exec grep -l "pub struct.*Model" {} \;`
- Verify all Model structs are either removed or use canonical domain types
- Ensure all imports reference fluent_ai_domain::model::*
- Confirm zero compilation errors related to domain duplication

## Quality Standards
- Zero allocation, blazing-fast, no locking, elegant ergonomic code
- Never use unwrap() or expect() in src
- Use Desktop Commander for all CLI commands
- All code must be production-ready with no TODOs or future enhancements
- Complete error handling with semantic error types

## Success Criteria
- 0 compilation errors workspace-wide
- 0 compilation warnings workspace-wide  
- ALL domain types imported from ./packages/domain/** only
- NO duplicated domain models anywhere in codebase