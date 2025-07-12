# TODO: Architectural Refactor - Fluent-AI-Rig Integration

## 1. Fix ARCHITECTURE.md Documentation
- [ ] Remove all formatting/rendering logic from chat closure example in ARCHITECTURE.md
- [ ] Show pure ChatLoop::Reprompt/ChatLoop::Break pattern with no formatting code
- [ ] Update example to demonstrate users only read conversation and return enum
- [ ] Replace current complex formatting example with simple enum-based control flow

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 2. QA: Architecture Documentation Compliance
- [ ] Act as an Objective QA Rust developer and verify ARCHITECTURE.md shows pure chat closure pattern with ChatLoop enum, no formatting logic, and demonstrates that all rendering is handled automatically by the builder behind the scenes.

## 3. Create ChatLoop Enum
- [ ] Define ChatLoop enum with Reprompt(String) and Break variants in fluent-ai/src/
- [ ] Integrate ChatLoop enum with existing .chat() method signature
- [ ] Ensure ChatLoop works with fluent-ai's async streaming architecture
- [ ] Export ChatLoop enum properly for use in fluent-ai-rig

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 4. QA: ChatLoop Enum Implementation
- [ ] Act as an Objective QA Rust developer and verify ChatLoop enum compiles, integrates with .chat() method, works with async streaming, and is properly exported for use in dependent crates.

## 5. Remove Direct Backend Calls from Main.rs (Critical Architectural Fix)
- [ ] Remove ALL direct calls to RigCompletionBackend from main.rs (current architectural violation)
- [ ] Remove direct backend.submit_completion() calls that bypass fluent-ai
- [ ] Remove manual completion handling and buffering logic
- [ ] Preserve CLI argument parsing but remove direct backend integration

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 6. QA: Direct Backend Call Removal
- [ ] Act as an Objective QA Rust developer and verify main.rs contains zero direct calls to rig-core or RigCompletionBackend, all backend interaction is removed, and CLI parsing remains intact.

## 7. Implement Fluent-AI Builder Pattern in Main.rs
- [ ] Map CLI --provider flag to .completion_provider() builder method
- [ ] Map CLI --model flag to model selection within provider using generated enums
- [ ] Map CLI --temperature flag to .temperature() builder method  
- [ ] Map CLI --agent-role flag to .agent_role() builder method
- [ ] Map CLI --context flag to .context() builder method with file/dir/glob loading

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 8. QA: Builder Pattern Implementation
- [ ] Act as an Objective QA Rust developer and verify main.rs uses proper fluent-ai builder pattern, all CLI flags map to builder methods, generated enums are used (no string matching), and the pattern follows fluent-ai architecture.

## 9. Implement .on_chunk() Real-Time Streaming
- [ ] Add .on_chunk() callback to builder chain for token-by-token streaming output
- [ ] Ensure streaming prints each token immediately (no buffering)
- [ ] Handle streaming errors gracefully without breaking chat session
- [ ] Integrate automatic markdown rendering for streamed tokens

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 10. QA: Real-Time Streaming Implementation
- [ ] Act as an Objective QA Rust developer and verify .on_chunk() provides real-time token streaming, handles errors gracefully, applies automatic formatting, and maintains proper chat session state.

## 11. Implement .chat() Closure with ChatLoop Enum
- [ ] Implement .chat() closure that uses ChatLoop::Reprompt/ChatLoop::Break pattern
- [ ] Ensure closure only reads conversation data and returns ChatLoop enum
- [ ] Remove any formatting, I/O, or manual loop logic from closure
- [ ] Implement conversation state management within closure

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 12. QA: Chat Closure Implementation
- [ ] Act as an Objective QA Rust developer and verify .chat() closure is pure, uses ChatLoop enum correctly, contains no formatting/I/O logic, and properly manages conversation state.

## 13. Integrate Automatic Markdown Rendering Behind the Scenes
- [ ] Integrate existing MarkdownRenderer from zeroshot codebase into fluent-ai builder
- [ ] Add pulldown_cmark dependency for markdown parsing
- [ ] Add syntect dependency for syntax highlighting
- [ ] Implement automatic formatting for agent responses (🤖 icons, markdown, code highlighting)
- [ ] Ensure all formatting happens in builder implementation, not user code

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 14. QA: Automatic Rendering Integration
- [ ] Act as an Objective QA Rust developer and verify MarkdownRenderer is integrated, dependencies are added, automatic formatting works, and no formatting logic exists in user-facing code.

## 15. Implement Context Loading (Files/Dirs/Globs/GitHub)
- [ ] Implement file reading with proper error handling and encoding detection
- [ ] Implement recursive directory traversal with file filtering
- [ ] Implement glob pattern matching for cross-platform compatibility  
- [ ] Implement GitHub reference loading with API integration
- [ ] Optimize context concatenation and LLM token limits

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 16. QA: Context Loading Implementation
- [ ] Act as an Objective QA Rust developer and verify context loading handles files/dirs/globs/GitHub refs correctly, includes proper error handling, and optimizes for token limits.

## 17. Verify RigCompletionBackend Integration with Fluent-AI
- [ ] Ensure RigCompletionBackend works seamlessly with fluent-ai's Engine system
- [ ] Verify backend can be selected via .completion_provider() method
- [ ] Test streaming integration between rig-core and fluent-ai
- [ ] Validate all provider support (openai, anthropic, mistral) through fluent-ai

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 18. QA: Backend Integration Verification
- [ ] Act as an Objective QA Rust developer and verify RigCompletionBackend integrates with fluent-ai Engine system, works via .completion_provider(), supports streaming, and handles all providers correctly.

## 19. End-to-End Testing and Production Quality Validation
- [ ] Test complete CLI workflow: `fluent_ai_rig --provider openai --model gpt-4o-mini "Hello"`
- [ ] Verify real-time streaming with beautiful automatic formatting
- [ ] Test all provider/model combinations with actual API keys
- [ ] Validate error handling for network failures, invalid keys, etc.
- [ ] Confirm zero compilation errors/warnings throughout codebase

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## 20. QA: End-to-End Production Quality Validation
- [ ] Act as an Objective QA Rust developer and verify complete CLI workflow functions correctly, streaming works in real-time, all providers are supported, error handling is comprehensive, and code quality meets production standards with zero warnings.