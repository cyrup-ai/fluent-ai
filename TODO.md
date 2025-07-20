# Fluent AI - Warnings and Errors Fix Log

This file tracks all warnings and errors in the codebase and their resolution status.

## Current Status
- **Errors**: 127
- **Warnings**: 11

## Errors to Fix

### 1. `packages/domain/src/chat/export.rs`
- [ ] `unwrap_or_else` not found for `std::time::Duration` (lines 234, 259, 289)
- [ ] No field `timestamp` on type `&LegacyMessage` (lines 376, 400, 433)

### 2. `packages/domain/src/chat/formatting.rs`
- [ ] Type annotations needed for `Arc<_, _>` (line 349)

### 3. `packages/domain/src/chat/macros.rs`
- [ ] Mismatched types: expected `Option<ExecutionStats>`, found `Option<&ExecutionStats>` (line 619)
- [ ] No method named `load` found for struct `ConsistentCounter` (lines 626, 631)
- [ ] Mismatched types: expected `AtomicUsize`, found `usize` (line 1300)

### 4. `packages/domain/src/agent/chat.rs`
- [ ] Binding modifiers not allowed under `ref` default binding mode (line 539)
- [ ] Borrow of moved value: `content` (line 194)

## Warnings to Fix

### 1. `packages/domain/src/agent/chat.rs`
- [ ] Unused variable: `message` (line 146)
- [ ] Unused variable: `memory_node` (line 147)

### 2. `packages/domain/src/chat/commands/types.rs`
- [ ] Unused variable: `command` (line 478)
- [ ] Unused variable: `context` (line 478)

### 3. `packages/domain/src/chat/templates.rs`
- [ ] Unused variable: `variable` (line 881)
- [ ] Unused variable: `variables` (line 1065)
- [ ] Unused variable: `args` (line 1070)
- [ ] Variable does not need to be mutable (line 1584)

### 4. `packages/domain/src/memory/manager.rs`
- [ ] Unused variable: `embedding_dim` (line 117)

### 5. `packages/domain/src/model/registry.rs`
- [ ] Unused variable: `handle` (lines 231, 265)

## Fixing Strategy
1. Start with the errors in `chat/export.rs` as they appear first in the compilation process
2. Move on to other errors in dependency order
3. Finally, address all warnings
4. After each fix, run `cargo check` to verify

## Fix Log

### 2025-07-20
- Initialized TODO.md with all current errors and warnings from `cargo check`
- Organized issues by file and type for systematic fixing