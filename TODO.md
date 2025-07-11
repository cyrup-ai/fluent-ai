# TODO.md - Typesafe Typestate Engine Builder Implementation

## Phase 1: Fix Core Engine Compilation Issues

### 1. Resolve async trait object safety compilation errors
- Fix async_trait macro usage for trait objects
- Ensure Send + Sync bounds are properly handled  
- Test trait object creation and usage
- Verify engine registry compilation
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 2. Act as an Objective QA Rust developer and rate the work performed on fixing engine trait compilation issues, ensuring async trait object safety is properly implemented and all bounds are correctly specified.

### 3. Validate engine registry functionality
- Test engine registration and retrieval
- Verify default engine setting and getting
- Ensure thread safety of global registry
- Test error handling for missing engines
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 4. Act as an Objective QA Rust developer and rate the work performed on engine registry validation, ensuring thread safety, proper error handling, and correct global state management.

## Phase 2: Typesafe Typestate Engine Builder Design

### 5. Design phantom type states for engine builder
- Create NeedsModel, NeedsConfiguration, Ready states
- Define state transition methods with phantom types
- Ensure compile-time validation of required fields
- Design immutable builder pattern with new instance returns
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 6. Act as an Objective QA Rust developer and rate the work performed on typestate builder design, ensuring proper phantom type usage, compile-time safety, and immutable patterns.

### 7. Implement EngineBuilder core structure
- Create EngineBuilder<State> generic struct
- Implement state marker traits (ModelState, ConfigState, etc.)
- Add phantom type field to builder struct
- Create initial builder constructor method
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 8. Act as an Objective QA Rust developer and rate the work performed on EngineBuilder core implementation, ensuring proper generic constraints and phantom type handling.

### 9. Implement model configuration methods

### 2. Missing Type Errors

- [ ] **Fix missing AgentConfig in domain::agent**
  - Error: `unresolved import crate::domain::agent::AgentConfig: no AgentConfig in domain::agent`
  - Locations: `engine.rs:9:35` and `fluent_engine.rs:8:35`
  - Fix: Locate actual AgentConfig definition or create if missing

- [ ] **QA: Rate AgentConfig resolution quality (1-10 scale)**
  - Verify AgentConfig exists and is properly exported
  - Confirm all AgentConfig usage compiles correctly
  - Check AgentConfig has all required fields and methods

- [ ] **Fix missing CompletionResponse in domain::completion**
  - Error: `unresolved import crate::domain::completion::CompletionResponse: no CompletionResponse in domain::completion`
  - Location: `fluent-ai/src/engine/fluent_engine.rs:9:52`
  - Fix: Locate actual CompletionResponse or create if missing

- [ ] **QA: Rate CompletionResponse resolution quality (1-10 scale)**
  - Verify CompletionResponse exists and is properly exported
  - Confirm all completion logic uses correct response type
  - Check response structure matches usage patterns

- [ ] **Fix missing NoOpAgent in domain::agent**
  - Error: `failed to resolve: could not find NoOpAgent in agent`
  - Location: `fluent-ai/src/engine.rs:198:47`
  - Fix: Locate actual NoOpAgent implementation or create if missing

- [ ] **QA: Rate NoOpAgent resolution quality (1-10 scale)**
  - Verify NoOpAgent implements required Agent trait
  - Confirm NoOpAgent provides appropriate no-op behavior
  - Check NoOpAgent is properly exported from agent module

### 3. Type Errors

- [ ] **Fix Agent trait vs struct confusion in fluent_engine.rs**
  - Error: `expected trait, found struct Agent: not a trait`
  - Locations: Lines 86, 93, 96 in `fluent-ai/src/engine/fluent_engine.rs`
  - Fix: Use correct Agent trait reference, not struct

- [ ] **QA: Rate Agent trait usage fix quality (1-10 scale)**
  - Verify Agent trait is used correctly in all contexts
  - Confirm no struct/trait confusion remains
  - Check all Agent implementations compile correctly

## Phase 2: Configuration Warnings (4 Total) ⚙️

- [ ] **Fix unexpected cfg condition pdf warnings**
  - Warning: `unexpected cfg condition value: pdf`
  - Locations: `loaders/mod.rs:18:7` and `loaders/mod.rs:21:7`
  - Fix: Add pdf feature to Cargo.toml or remove unused cfg conditions

- [ ] **QA: Rate pdf cfg condition fix quality (1-10 scale)**
  - Verify pdf feature is properly configured if needed
  - Confirm no unused conditional compilation remains
  - Check feature flags align with actual usage

- [ ] **Fix unexpected cfg condition epub warnings**
  - Warning: `unexpected cfg condition value: epub`
  - Locations: `loaders/mod.rs:24:7` and `loaders/mod.rs:27:7`
  - Fix: Add epub feature to Cargo.toml or remove unused cfg conditions

- [ ] **QA: Rate epub cfg condition fix quality (1-10 scale)**
  - Verify epub feature is properly configured if needed
  - Confirm conditional compilation logic is correct
  - Check feature dependencies are properly specified

## Phase 3: Unused Import Warnings (22 Total) 🧹

### Domain Module Unused Imports

- [ ] **Fix unused ChatMessageChunk import in completion.rs:3:28**
  - Warning: `unused import: ChatMessageChunk`
  - Fix: Remove import or implement usage as intended

- [ ] **QA: Rate ChatMessageChunk fix quality (1-10 scale)**
  - Verify removal doesn't break intended functionality
  - Confirm no missing implementation was intended
  - Check import cleanup follows project conventions

- [ ] **Fix unused CompletionRequest and Message imports in extractor.rs:4:21**
  - Warning: `unused imports: CompletionRequest and Message`
  - Fix: Remove imports or implement extraction functionality

- [ ] **QA: Rate extractor imports fix quality (1-10 scale)**
  - Verify extractor functionality is complete or properly stubbed
  - Confirm no missing extractor features
  - Check extractor module serves its intended purpose

- [ ] **Fix unused AsyncTask import in image.rs:2:26**
  - Warning: `unused import: AsyncTask`
  - Fix: Remove import or implement async image processing
### 28. Act as an Objective QA Rust developer and rate the work performed on documentation, ensuring completeness, clarity, and practical examples for users.

### 29. Implement comprehensive test suite
- Add unit tests for each builder state and method
- Create integration tests with engine registry
- Test error conditions and validation
- Add property-based tests for builder invariants
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 30. Act as an Objective QA Rust developer and rate the work performed on the test suite, ensuring comprehensive coverage, proper test organization, and verification of all builder functionality.
