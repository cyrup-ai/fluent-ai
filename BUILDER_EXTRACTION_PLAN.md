# BUILDER_EXTRACTION_PLAN.md - Complete Domain/Builder Separation

## Overview
Complete architectural separation between domain and fluent_ai packages:
- **Domain**: Pure traits, structs (data objects), enums only
- **Fluent_ai**: All builders, implementations, execution logic

## Phase 1: Architecture Audit and Planning

### Task 1: Audit domain/src/agent.rs for builder separation
- **File**: `packages/domain/src/agent.rs`
- **Action**: Identify AgentBuilder and any implementation types that need to move to fluent_ai
- **Architecture**: Domain should only contain Agent trait and data structures, no builders
- **Implementation**: Scan for struct definitions ending in "Builder" or "Impl"
- **Expected**: Clear list of what stays in domain vs moves to fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 2: QA Task - Agent Module Audit
Act as an Objective QA Rust developer and rate the agent module audit on a scale of 1-10. Verify that all builders and implementations were correctly identified for migration.

### Task 3: Audit domain/src/audio.rs for builder separation
- **File**: `packages/domain/src/audio.rs`
- **Action**: Identify AudioBuilder, AudioBuilderWithHandler and any implementation types
- **Architecture**: Domain should only contain Audio struct and traits, no builders
- **Implementation**: Scan for builder patterns and implementation logic
- **Expected**: Clear separation between data structures and behavior
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 4: QA Task - Audio Module Audit
Act as an Objective QA Rust developer and rate the audio module audit on a scale of 1-10. Verify that data structures are properly separated from builders.

### Task 5: Audit domain/src/completion.rs for builder separation
- **File**: `packages/domain/src/completion.rs`
- **Action**: Identify CompletionRequestBuilder, CompletionRequestBuilderWithHandler
- **Architecture**: Domain should only contain CompletionRequest struct and traits
- **Implementation**: Separate builder logic from data definitions
- **Expected**: Clean data layer without construction logic
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 6: QA Task - Completion Module Audit
Act as an Objective QA Rust developer and rate the completion module audit on a scale of 1-10. Verify proper separation of concerns.

### Task 7: Audit domain/src/conversation.rs for builder separation
- **File**: `packages/domain/src/conversation.rs`
- **Action**: Identify ConversationBuilder, ConversationBuilderWithHandler, ConversationImpl
- **Architecture**: Domain should only contain Conversation trait, no implementations or builders
- **Implementation**: Move all concrete implementations to fluent_ai
- **Expected**: Pure trait definitions in domain
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 8: QA Task - Conversation Module Audit
Act as an Objective QA Rust developer and rate the conversation module audit on a scale of 1-10. Verify trait/implementation separation.

### Task 9: Audit domain/src/image.rs for builder separation
- **File**: `packages/domain/src/image.rs`
- **Action**: Identify ImageBuilder, ImageBuilderWithHandler
- **Architecture**: Domain should only contain Image struct and data types
- **Implementation**: Separate construction logic from data representation
- **Expected**: Pure data structures without builder patterns
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 10: QA Task - Image Module Audit
Act as an Objective QA Rust developer and rate the image module audit on a scale of 1-10. Verify clean data layer separation.

### Task 11: Audit domain/src/mcp.rs for builder separation
- **File**: `packages/domain/src/mcp.rs`
- **Action**: Identify McpClientBuilder and any implementation types
- **Architecture**: Domain should only contain Client trait and data structures
- **Implementation**: Move builder logic to fluent_ai
- **Expected**: Pure protocol definitions without construction logic
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 12: QA Task - MCP Module Audit
Act as an Objective QA Rust developer and rate the MCP module audit on a scale of 1-10. Verify protocol/implementation separation.

### Task 13: Audit domain/src/message.rs for builder separation
- **File**: `packages/domain/src/message.rs`
- **Action**: Identify MessageBuilder, UserMessageBuilderTrait, AssistantMessageBuilderTrait, MessageFactory
- **Architecture**: Domain should only contain Message struct and data types
- **Implementation**: Move all builder traits and factory patterns to fluent_ai
- **Expected**: Pure message data structures without construction logic
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 14: QA Task - Message Module Audit
Act as an Objective QA Rust developer and rate the message module audit on a scale of 1-10. Verify data/behavior separation.

### Task 15: Audit domain/src/prompt.rs for builder separation
- **File**: `packages/domain/src/prompt.rs`
- **Action**: Identify PromptBuilder and any implementation types
- **Architecture**: Domain should only contain Prompt struct and data types
- **Implementation**: Move builder logic to fluent_ai
- **Expected**: Pure prompt data without construction patterns
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 16: QA Task - Prompt Module Audit
Act as an Objective QA Rust developer and rate the prompt module audit on a scale of 1-10. Verify clean separation achieved.

## Phase 2: Builder Migration to Fluent_ai

### Task 17: Create fluent_ai/src/builders/ directory structure
- **File**: `packages/fluent_ai/src/builders/mod.rs`
- **Action**: Create builders module with proper structure
- **Architecture**: Centralized location for all builder implementations
- **Implementation**: Create mod.rs with re-exports for all builder modules
- **Expected**: Clean module organization for builders
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 18: QA Task - Builder Directory Structure
Act as an Objective QA Rust developer and rate the builder directory structure on a scale of 1-10. Verify proper organization.

### Task 19: Move AgentBuilder to fluent_ai/src/builders/agent.rs
- **File**: `packages/fluent_ai/src/builders/agent.rs`
- **Action**: Move AgentBuilder from domain to fluent_ai
- **Architecture**: Builder implementations belong in fluent_ai, not domain
- **Implementation**: Copy builder code, update imports, remove from domain
- **Expected**: AgentBuilder working from fluent_ai location
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 20: QA Task - Agent Builder Migration
Act as an Objective QA Rust developer and rate the agent builder migration on a scale of 1-10. Verify builder works correctly from new location.

### Task 21: Move AudioBuilder to fluent_ai/src/builders/audio.rs
- **File**: `packages/fluent_ai/src/builders/audio.rs`
- **Action**: Move AudioBuilder, AudioBuilderWithHandler from domain to fluent_ai
- **Architecture**: Audio construction logic belongs in fluent_ai
- **Implementation**: Migrate builder implementations with proper imports
- **Expected**: Audio builders working from fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 22: QA Task - Audio Builder Migration
Act as an Objective QA Rust developer and rate the audio builder migration on a scale of 1-10. Verify builders function correctly.

### Task 23: Move CompletionRequestBuilder to fluent_ai/src/builders/completion.rs
- **File**: `packages/fluent_ai/src/builders/completion.rs`
- **Action**: Move CompletionRequestBuilder, CompletionRequestBuilderWithHandler from domain
- **Architecture**: Completion construction logic belongs in fluent_ai
- **Implementation**: Migrate builder code with updated imports
- **Expected**: Completion builders working from fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 24: QA Task - Completion Builder Migration
Act as an Objective QA Rust developer and rate the completion builder migration on a scale of 1-10. Verify proper functionality.

### Task 25: Move ConversationBuilder to fluent_ai/src/builders/conversation.rs
- **File**: `packages/fluent_ai/src/builders/conversation.rs`
- **Action**: Move ConversationBuilder, ConversationBuilderWithHandler, ConversationImpl from domain
- **Architecture**: Conversation implementations belong in fluent_ai
- **Implementation**: Migrate builders and implementations with proper imports
- **Expected**: Conversation builders working from fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 26: QA Task - Conversation Builder Migration
Act as an Objective QA Rust developer and rate the conversation builder migration on a scale of 1-10. Verify complete functionality.

### Task 27: Move ImageBuilder to fluent_ai/src/builders/image.rs
- **File**: `packages/fluent_ai/src/builders/image.rs`
- **Action**: Move ImageBuilder, ImageBuilderWithHandler from domain to fluent_ai
- **Architecture**: Image construction logic belongs in fluent_ai
- **Implementation**: Migrate builder implementations with updated imports
- **Expected**: Image builders working from fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 28: QA Task - Image Builder Migration
Act as an Objective QA Rust developer and rate the image builder migration on a scale of 1-10. Verify builders work correctly.

### Task 29: Move McpClientBuilder to fluent_ai/src/builders/mcp.rs
- **File**: `packages/fluent_ai/src/builders/mcp.rs`
- **Action**: Move McpClientBuilder from domain to fluent_ai
- **Architecture**: MCP client construction belongs in fluent_ai
- **Implementation**: Migrate builder code with proper imports
- **Expected**: MCP client builder working from fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 30: QA Task - MCP Builder Migration
Act as an Objective QA Rust developer and rate the MCP builder migration on a scale of 1-10. Verify builder functionality.

### Task 31: Move MessageBuilder to fluent_ai/src/builders/message.rs
- **File**: `packages/fluent_ai/src/builders/message.rs`
- **Action**: Move MessageBuilder, UserMessageBuilderTrait, AssistantMessageBuilderTrait, MessageFactory
- **Architecture**: Message construction logic belongs in fluent_ai
- **Implementation**: Migrate all builder traits and factory patterns
- **Expected**: Message builders working from fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 32: QA Task - Message Builder Migration
Act as an Objective QA Rust developer and rate the message builder migration on a scale of 1-10. Verify complete builder functionality.

### Task 33: Move PromptBuilder to fluent_ai/src/builders/prompt.rs
- **File**: `packages/fluent_ai/src/builders/prompt.rs`
- **Action**: Move PromptBuilder from domain to fluent_ai
- **Architecture**: Prompt construction logic belongs in fluent_ai
- **Implementation**: Migrate builder code with updated imports
- **Expected**: Prompt builder working from fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 34: QA Task - Prompt Builder Migration
Act as an Objective QA Rust developer and rate the prompt builder migration on a scale of 1-10. Verify builder works correctly.

## Phase 3: Re-export and Import Cleanup

### Task 35: Remove builder re-exports from domain/src/lib.rs
- **File**: `packages/domain/src/lib.rs`
- **Action**: Remove all builder re-exports (AgentBuilder, AudioBuilder, etc.)
- **Architecture**: Domain should not expose builder types
- **Implementation**: Delete builder re-export lines from lib.rs
- **Expected**: Domain lib.rs contains only data types and traits
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 36: QA Task - Domain Re-export Cleanup
Act as an Objective QA Rust developer and rate the domain re-export cleanup on a scale of 1-10. Verify no builders are exposed.

### Task 37: Add builder re-exports to fluent_ai/src/lib.rs
- **File**: `packages/fluent_ai/src/lib.rs`
- **Action**: Add re-exports for all moved builders
- **Architecture**: Fluent_ai should expose builder types for user consumption
- **Implementation**: Add pub use statements for builders module
- **Expected**: Builders accessible from fluent_ai crate root
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 38: QA Task - Fluent_ai Re-export Addition
Act as an Objective QA Rust developer and rate the fluent_ai re-export addition on a scale of 1-10. Verify builders are properly exposed.

### Task 39: Update fluent_ai/src/lib.rs builders module reference
- **File**: `packages/fluent_ai/src/lib.rs`
- **Action**: Add `pub mod builders;` and re-export builders module
- **Architecture**: Proper module organization in fluent_ai
- **Implementation**: Add module declaration and re-exports
- **Expected**: Builders module properly integrated into fluent_ai
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 40: QA Task - Builders Module Integration
Act as an Objective QA Rust developer and rate the builders module integration on a scale of 1-10. Verify proper module structure.

## Phase 4: Domain Cleanup and Purification

### Task 41: Remove cylo feature flags from domain/src/tool.rs
- **File**: `packages/domain/src/tool.rs`
- **Action**: Remove all `#[cfg(feature = "cylo")]` directives
- **Architecture**: Domain should not have execution-specific feature flags
- **Implementation**: Remove conditional compilation blocks
- **Expected**: Clean domain code without execution dependencies
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 42: QA Task - Domain Feature Flag Cleanup
Act as an Objective QA Rust developer and rate the domain feature flag cleanup on a scale of 1-10. Verify no execution dependencies remain.

### Task 43: Fix McpTool name conflicts in domain/src/lib.rs
- **File**: `packages/domain/src/lib.rs:288-292`
- **Action**: Remove duplicate McpTool import that causes name conflict
- **Architecture**: Clean import organization in domain
- **Implementation**: Keep only one McpTool import from mcp_tool_traits
- **Expected**: No name conflicts in domain re-exports
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 44: QA Task - McpTool Name Conflict Resolution
Act as an Objective QA Rust developer and rate the McpTool name conflict resolution on a scale of 1-10. Verify clean imports.

### Task 45: Remove McpToolImpl import from domain/src/agent.rs
- **File**: `packages/domain/src/agent.rs:7`
- **Action**: Remove import of McpToolImpl since it's now in fluent_ai
- **Architecture**: Domain should not import implementation types
- **Implementation**: Remove McpToolImpl from import statement
- **Expected**: Domain agent module with clean imports
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 46: QA Task - Agent Module Import Cleanup
Act as an Objective QA Rust developer and rate the agent module import cleanup on a scale of 1-10. Verify no implementation imports.

### Task 47: Clean up all domain modules to remove builder references
- **Files**: All `packages/domain/src/*.rs` files
- **Action**: Remove any remaining builder imports or references
- **Architecture**: Domain modules should be pure data/trait definitions
- **Implementation**: Scan and remove builder-related code
- **Expected**: Clean domain modules without builder logic
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 48: QA Task - Domain Module Purification
Act as an Objective QA Rust developer and rate the domain module purification on a scale of 1-10. Verify complete separation achieved.

## Phase 5: Compilation and Integration Testing

### Task 49: Test domain package compilation
- **Command**: `cargo check --package fluent_ai_domain`
- **Action**: Verify domain package compiles cleanly
- **Architecture**: Domain should compile without fluent_ai dependencies
- **Implementation**: Run compilation check and fix any issues
- **Expected**: Clean compilation with 0 errors and 0 warnings
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 50: QA Task - Domain Package Compilation
Act as an Objective QA Rust developer and rate the domain package compilation on a scale of 1-10. Verify clean compilation.

### Task 51: Test fluent_ai package compilation
- **Command**: `cargo check --package fluent_ai`
- **Action**: Verify fluent_ai package compiles with builders
- **Architecture**: Fluent_ai should compile with all builder implementations
- **Implementation**: Run compilation check and fix any issues
- **Expected**: Clean compilation with 0 errors and 0 warnings
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 52: QA Task - Fluent_ai Package Compilation
Act as an Objective QA Rust developer and rate the fluent_ai package compilation on a scale of 1-10. Verify successful compilation.

### Task 53: Test full workspace compilation
- **Command**: `cargo check`
- **Action**: Verify entire workspace compiles cleanly
- **Architecture**: All packages should work together with clean separation
- **Implementation**: Run full workspace compilation check
- **Expected**: Complete workspace compilation with 0 errors and 0 warnings
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 54: QA Task - Full Workspace Compilation
Act as an Objective QA Rust developer and rate the full workspace compilation on a scale of 1-10. Verify complete integration success.

### Task 55: Test builder functionality from fluent_ai
- **Files**: All builder modules in `packages/fluent_ai/src/builders/`
- **Action**: Verify builders work correctly from new location
- **Architecture**: Builders should function properly from fluent_ai
- **Implementation**: Test builder usage patterns
- **Expected**: All builders working as expected
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 56: QA Task - Builder Functionality Verification
Act as an Objective QA Rust developer and rate the builder functionality verification on a scale of 1-10. Verify all builders work correctly.

## Final Verification

### Task 57: Verify architectural boundaries are maintained
- **Action**: Confirm domain contains only traits, structs, enums
- **Architecture**: Clean separation between data layer and behavior layer
- **Implementation**: Review all domain modules for architectural compliance
- **Expected**: Perfect architectural separation achieved
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task 58: QA Task - Architectural Boundary Verification
Act as an Objective QA Rust developer and rate the architectural boundary verification on a scale of 1-10. Verify clean separation maintained.