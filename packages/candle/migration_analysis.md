# Chat System Migration Analysis

## OVERVIEW
Analysis of current import dependencies across both chat systems for migration planning.

## DUAL IMPORT CONFLICTS (HIGH PRIORITY)

### Critical Conflicts Requiring Immediate Resolution
1. **./src/agent/agent.rs** (Lines 15-16)
   - Imports: `crate::domain::chat::message::types` (Domain types)  
   - Imports: `crate::chat::ChatLoop` (Original ChatLoop)
   - **CONFLICT**: Uses types from domain system but ChatLoop from original system

2. **./src/domain/chat/search/mod.rs** (Lines 27-29, 31)
   - Imports: `crate::chat::` (Original system - lines 27-29)
   - Imports: `crate::domain::chat::` (Domain system - line 31)
   - **CONFLICT**: Mixed imports from both systems in same file

## ORIGINAL CHAT SYSTEM IMPORTS (./src/chat/)

### Files Importing from Original System
- `./src/agent/agent.rs:16` - `use crate::chat::`
- `./src/chat/search/mod.rs:31` - `use crate::chat::`
- `./src/chat/templates/filters.rs:7` - `use crate::chat::`
- `./src/chat/macros.rs:20` - `use crate::chat::`
- `./src/chat/search/manager/mod.rs:5` - `use crate::chat::`
- `./src/chat/search/types.rs:9` - `use crate::chat::`
- `./src/chat/templates/compiler.rs:5` - `use crate::chat::`
- `./src/chat/search/index.rs:15` - `use crate::chat::`
- `./src/chat/templates/engines.rs:7` - `use crate::chat::`
- `./src/chat/templates/manager.rs:9` - `use crate::chat::`
- `./src/chat/templates/cache/store.rs:8` - `use crate::chat::`

### Original System Internal Structure
Most imports are within ./src/chat/ itself (internal dependencies), except:
- **EXTERNAL**: `./src/agent/agent.rs` - imports ChatLoop from original system
- **MIXED**: `./src/domain/chat/search/mod.rs` - imports from both systems

## DOMAIN CHAT SYSTEM IMPORTS (./src/domain/chat/)

### Builder System Integration
- `./src/prompt/mod.rs:3` - `use crate::domain::chat::`
- `./src/agent/role.rs:13` - `use crate::domain::chat::`
- `./src/domain/agent/types.rs:7` - `use crate::domain::chat::`
- `./src/agent/agent.rs:15` - `use crate::domain::chat::`
- `./src/agent/types.rs:6` - `use crate::domain::chat::`
- `./src/domain/agent/role.rs:13` - `use crate::domain::chat::`
- `./src/domain/prompt/mod.rs:3` - `use crate::domain::chat::`
- `./src/domain/completion/request.rs:15` - `use crate::domain::chat::`

### Builder Pattern Integration
- `./src/builders/message.rs:5` - `use crate::domain::chat::`
- `./src/builders/live_message_streamer.rs:5` - `use crate::domain::chat::`
- `./src/builders/typing_indicator.rs:12` - `use crate::domain::chat::`
- `./src/builders/realtime_system.rs:9` - `use crate::domain::chat::`
- `./src/builders/chat_config.rs:7` - `use crate::domain::chat::`
- `./src/builders/chat/conversation_builder.rs:2-3` - `use crate::domain::chat::`
- `./src/builders/chat/history_manager_builder.rs:4` - `use crate::domain::chat::`
- `./src/builders/chat/template_builder.rs:3,7` - `use crate::domain::chat::`
- `./src/builders/chat/macro_builder.rs:6` - `use crate::domain::chat::`

### Core Library Integration
- `./src/lib.rs:36` - `use crate::domain::chat::`

### Cross-System References
- `./src/chat/commands/execution.rs:14` - `use crate::domain::chat::`
- `./src/chat/commands/types/mod.rs:4` - `use crate::domain::chat::`
- `./src/chat/realtime/streaming.rs:22` - `use crate::domain::chat::`
- `./src/chat/realtime/mod.rs:27` - `use crate::domain::chat::`
- `./src/chat/realtime/events.rs:10` - `use crate::domain::chat::`

## TYPE MAPPING ANALYSIS

### Domain System Types (Candle-prefixed)
- `CandleMessageRole` (MessageRole)
- `CandleMessageChunk` (MessageChunk)  
- `CandleConversationTrait` (AgentConversation)
- `CandleZeroOneOrMany` (ZeroOneOrMany)

### Original System Types (Non-prefixed)
- `ChatLoop` - **CRITICAL**: Imported by agent.rs
- `Message` - Search and template types
- Various search/filter/template types

## MIGRATION DEPENDENCIES

### Phase 1 Priority (Must Resolve First)
1. **ChatLoop Migration**: Replace `crate::chat::ChatLoop` with `crate::domain::chat::CandleChatLoop`
2. **Mixed Imports**: Fix ./src/domain/chat/search/mod.rs dual imports
3. **Agent Integration**: Update ./src/agent/agent.rs to use only domain system

### Phase 2 Dependencies (Original System Internal)
All ./src/chat/ internal imports will be automatically resolved when system is removed.

### Phase 3 Dependencies (Cross-System References)
Files in ./src/chat/ importing from ./src/domain/chat/ need careful migration to ensure functionality is preserved.

## ARCHITECTURAL NOTES

### Connection Points
- **Agent System**: Primary connection point between both systems
- **Builder Pattern**: Exclusively uses domain system (correct)
- **Realtime Features**: Original system imports domain types (indicates domain is authoritative)
- **Commands**: Original system imports domain types (indicates domain is authoritative)

### Migration Strategy
1. **Domain system is the target** (connected to builders, lib.rs, agents)
2. **Original system has superior features** (search, filtering, management)
3. **Cross-references indicate** domain system is the "live" system
4. **Builder integration confirms** domain system is the user-facing API

## COMPLETION CRITERIA
- [ ] Zero imports of `use crate::chat::` outside ./src/chat/ directory
- [ ] All ./src/chat/ references migrated to ./src/domain/chat/
- [ ] ./src/agent/agent.rs uses only domain system
- [ ] ./src/domain/chat/search/mod.rs resolved dual imports
- [ ] All type references use Candle-prefixed names consistently