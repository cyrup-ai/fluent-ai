# DOMAIN PACKAGE ERROR AND WARNING FIXES

## OBJECTIVE: 0 ERRORS, 0 WARNINGS

Current Status: ~87 errors, ~5 warnings (from latest cargo check)

## ERRORS TO FIX

### Category A: Missing Functions/Methods
1. `chat/commands/types.rs:494` - cannot find function `async_stream_channel` in scope
2. `chat/commands/types.rs:761` - no function `parse_template_command` found for CommandParser
3. `chat/commands/types.rs:762` - no function `parse_macro_command` found for CommandParser  
4. `chat/commands/types.rs:763` - no function `parse_search_command` found for CommandParser
5. `chat/commands/types.rs:764` - no function `parse_branch_command` found for CommandParser
6. `chat/commands/types.rs:765` - no function `parse_session_command` found for CommandParser
7. `chat/commands/types.rs:766` - no function `parse_tool_command` found for CommandParser
8. `chat/commands/types.rs:767` - no function `parse_stats_command` found for CommandParser
9. `chat/commands/types.rs:768` - no function `parse_theme_command` found for CommandParser
10. `chat/commands/mod.rs:61` - no method `execute` found for CommandExecutor
11. `chat/commands/mod.rs:80` - no method `wait` found for AsyncStream
12. `chat/config.rs:237` - cannot find function `spawn_stream` in scope

### Category B: Type Mismatches
13. `chat/commands/types.rs:475` - AsyncStreamSender<CommandEvent> doesn't implement Debug
14. `chat/commands/execution.rs:89` - expected AsyncStream<CommandOutput>, found UnboundedReceiverStream
15. `chat/commands/validation.rs:110` - expected &HashMap<Arc<str>, Arc<str>>, found &HashMap<String, String>
16. `chat/commands/validation.rs:134` - expected &HashMap<Arc<str>, Arc<str>>, found &HashMap<String, String>
17. `chat/commands/validation.rs:147` - expected &HashMap<Arc<str>, Arc<str>>, found &HashMap<String, String>
18. `memory/manager.rs:415` - expected AsyncStream<Memory>, found UnboundedReceiverStream<Memory>

### Category C: Struct Field Issues
19. `chat/commands/response.rs:272` - missing fields `content`, `execution_id`, `is_final` in CommandOutput initializer
20. `chat/commands/response.rs:274` - expected String, found Arc<_, _>
21. `chat/commands/response.rs:277` - expected Option<ResourceUsage>, found ResourceUsage
22. `chat/commands/response.rs:305` - expected Arc<str>, found String
23. `chat/commands/mod.rs:83` - variant CommandError::ExecutionFailed has no field `reason`
24. `context/provider.rs:503` - struct Document has no field `id`
25. `context/provider.rs:504` - struct Document has no field `content`
26. `context/provider.rs:505` - struct Document has no field `metadata`
27. `context/provider.rs:512` - struct Document has no field `embedding`

### Category D: Enum Variant Issues
28. `chat/commands/mod.rs:51` - expected value, found struct variant CommandError::ConfigurationError
29. `chat/commands/mod.rs:69` - expected value, found struct variant CommandError::ConfigurationError
30. `chat/commands/mod.rs:87` - expected value, found struct variant CommandError::ConfigurationError

### Category E: Function Argument Issues
31. `chat/commands/mod.rs:32` - function takes 0 arguments but 1 argument was supplied
32. `chat/commands/mod.rs:40` - method `clone` exists but trait bounds not satisfied

### Category F: Lifetime/Borrowing Issues
33. `completion/request.rs:169` - borrowed data escapes outside method body
34. `completion/request.rs:193` - borrowed data escapes outside method body  
35. `completion/request.rs:212` - borrowed data escapes outside method body
36. `completion/response.rs:296` - lifetime may not live long enough

### Category G: Async/Await Issues
37. `engine.rs:312` - expected FnOnce() closure, found async block
38. `engine.rs:345` - expected FnOnce() closure, found async block
39. `core/mod.rs:70` - expected Receiver<Result<T, ChannelError>>, found async block
40. `core/mod.rs:70` - T cannot be sent between threads safely

### Category H: Additional Type Issues
41. `embedding/core.rs:33` - expected ZeroOneOrMany<f32>, found Result<_, _>
42. `embedding/core.rs:37` - expected ZeroOneOrMany<f32>, found Result<_, _>
43. `context/provider.rs:417` - expected UnboundedReceiverStream, found AsyncStream<Result<Document, ...>>

## WARNINGS TO FIX

### Category W: Unused Imports
44. `agent/builder.rs:8` - unused import: `tokio_stream::StreamExt`
45. `chat/config.rs:16` - unused import: `tokio_stream::StreamExt`

### Category W: Unused Variables
46. `context/provider.rs:672` - unused variable: `files_context`
47. `context/provider.rs:712` - unused variable: `directory_context`
48. `context/provider.rs:752` - unused variable: `github_context`

## SYSTEMATIC FIX PLAN

### Phase 1: Quick Wins (Unused imports/variables)
- Fix unused import warnings
- Implement or remove unused variables

### Phase 2: Missing Function Implementation
- Implement all missing parse_* methods in CommandParser
- Add missing async_stream_channel function
- Add missing execute method to CommandExecutor
- Add missing spawn_stream function

### Phase 3: Type System Fixes
- Fix all HashMap<String, String> vs HashMap<Arc<str>, Arc<str>> mismatches
- Fix all AsyncStream vs UnboundedReceiverStream mismatches
- Add Debug implementation for AsyncStreamSender

### Phase 4: Struct Field Fixes
- Add missing fields to CommandOutput initialization
- Fix Document struct field issues
- Fix CommandError variant field issues

### Phase 5: Enum Variant Usage Fixes
- Fix all struct variant usage patterns

### Phase 6: Lifetime/Borrowing Fixes
- Fix all lifetime issues in completion modules

### Phase 7: Async/Await Cleanup
- Remove async/await from non-async contexts

## SUCCESS CRITERIA
- `cargo check` returns 0 errors, 0 warnings
- All code is production quality
- All fixes maintain zero allocation, no locking constraints
- End-user testing confirms functionality works