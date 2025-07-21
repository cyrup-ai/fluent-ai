# TODO: Fix Provider Build Script Dynamic Code Generation

## CURRENT WORKSPACE STATUS
- **ROOT CAUSE IDENTIFIED**: Provider build.rs is generating empty stub files instead of using the sophisticated code generation system
- **ERRORS**: 600+ compilation errors caused by missing generated provider and model code
- **OBJECTIVE**: Fix build script to generate proper provider/model code compatible with canonical domain types

## CRITICAL BUILD SCRIPT FIXES

### 1. Replace Stub build.rs with Full Code Generation System ⭐ CRITICAL
- **File**: `packages/provider/build.rs:1-27`
- **Issue**: Current build.rs only generates empty stub files, ignoring the complete code generation infrastructure
- **Architecture**: The build/ directory contains a complete code generation system (CodeGenerator, YamlProcessor, templates) that should be used
- **Implementation**:
  - Replace minimal build.rs with full integration of build/code_generator.rs
  - Use YamlProcessor to parse provider/model definitions from external YAML sources
  - Generate provider enums using build/templates/provider_struct.rs.template
  - Generate model enums using build/templates/model_enum.rs.template
  - Generate client implementations using build/templates/client_impl.rs.template
  - Ensure all generated code uses canonical fluent_ai_domain::model::ModelInfo types
- **Files to Modify**:
  - `packages/provider/build.rs` (lines 1-27) - Complete rewrite to use code generation system
  - Integration with `packages/provider/build/code_generator.rs` (lines 1-32)
  - Integration with `packages/provider/build/yaml_processor.rs` (lines 1-319)
- **Priority**: CRITICAL - This single fix will resolve 600+ compilation errors
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 2. QA: Build Script Integration Quality Assessment
Act as an Objective QA Rust developer and rate the work performed previously on replacing the stub build.rs with full code generation system integration. Evaluate: (1) Proper integration with existing CodeGenerator and YamlProcessor, (2) Correct usage of templates for provider/model generation, (3) Compatibility with canonical domain types, (4) Zero-allocation patterns maintained, (5) Error handling completeness. Rate 1-10 and provide specific feedback on any issues or excellent work. Any score below 9 requires rework.

### 3. Restore Missing CodeGenerator Methods ⭐ HIGH
- **File**: `packages/provider/build/code_generator.rs:1-32`
- **Issue**: All generation methods have been removed, leaving only an empty struct
- **Architecture**: CodeGenerator should have methods to generate provider modules, model registries, and client implementations
- **Implementation**:
  - Restore `generate_provider_module()` method to create provider enum with all variants
  - Restore `generate_model_registry()` method to create model definitions using canonical ModelInfo
  - Add `generate_client_implementations()` method for dynamic client creation
  - Integrate with template system in build/templates/
  - Ensure all generated code uses fluent_ai_domain::model::ModelInfo instead of local duplicates
- **Files to Modify**:
  - `packages/provider/build/code_generator.rs` (lines 1-32) - Restore removed methods
  - Integration with templates in `packages/provider/build/templates/`
- **Priority**: HIGH - Required for build script to function
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 4. QA: CodeGenerator Methods Restoration Quality Assessment
Act as an Objective QA Rust developer and rate the work performed previously on restoring CodeGenerator methods. Evaluate: (1) Method signatures match expected template system integration, (2) Generated code uses canonical domain types correctly, (3) Zero-allocation patterns maintained, (4) Template integration works properly, (5) Error handling is comprehensive. Rate 1-10 and provide specific feedback. Any score below 9 requires rework.

### 5. Fix Template Compatibility with Canonical Domain Types ⭐ HIGH
- **File**: `packages/provider/build/templates/*.rs.template`
- **Issue**: Templates likely reference old/incorrect type names causing generation failures
- **Architecture**: All templates must generate code that imports and uses fluent_ai_domain::model::ModelInfo
- **Implementation**:
  - Update `model_enum.rs.template` to use canonical ModelInfo from domain package
  - Update `provider_struct.rs.template` to import fluent_ai_domain types
  - Update `client_impl.rs.template` to use proper domain types in generated implementations
  - Update `imports.rs.template` to include canonical domain imports
  - Ensure `trait_impls.rs.template` generates compatible trait implementations
  - Fix `file_header.rs.template` to include proper module documentation
- **Files to Modify**:
  - `packages/provider/build/templates/model_enum.rs.template` - Use canonical ModelInfo
  - `packages/provider/build/templates/provider_struct.rs.template` - Fix domain imports
  - `packages/provider/build/templates/client_impl.rs.template` - Fix type references
  - `packages/provider/build/templates/imports.rs.template` - Add domain imports
  - `packages/provider/build/templates/trait_impls.rs.template` - Fix trait compatibility
  - `packages/provider/build/templates/file_header.rs.template` - Update documentation
- **Priority**: HIGH - Templates must generate compatible code
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 6. QA: Template Compatibility Quality Assessment
Act as an Objective QA Rust developer and rate the work performed previously on fixing template compatibility with canonical domain types. Evaluate: (1) All templates use correct fluent_ai_domain imports, (2) Generated code will compile without type errors, (3) Template variables are properly substituted, (4) Generated trait implementations are compatible, (5) Documentation generation is correct. Rate 1-10 and provide specific feedback. Any score below 9 requires rework.

### 7. Integrate External Model Data Source ⭐ HIGH
- **File**: `packages/provider/build.rs` (new implementation)
- **Issue**: Build script needs to fetch actual model data from external source (sigoden's models.yaml)
- **Architecture**: Use YamlProcessor to parse external model definitions and generate compatible code
- **Implementation**:
  - Add HTTP client to fetch models.yaml from sigoden's GitHub repository
  - Use YamlProcessor to parse the external YAML into ProviderInfo/ModelInfo structs
  - Convert external model format to canonical fluent_ai_domain::model::ModelInfo format
  - Generate provider enums with actual provider variants (OpenAI, Anthropic, Google, etc.)
  - Generate model definitions with real model capabilities and limits
  - Cache downloaded YAML to avoid repeated network requests during builds
- **Files to Modify**:
  - `packages/provider/build.rs` - Add external data fetching and processing
  - Integration with `packages/provider/build/yaml_processor.rs` for parsing
  - Use `packages/provider/build/http_client.rs` for downloading model data
- **Priority**: HIGH - Required for generating actual provider/model code
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 8. QA: External Model Data Integration Quality Assessment
Act as an Objective QA Rust developer and rate the work performed previously on integrating external model data source. Evaluate: (1) HTTP client properly fetches sigoden's models.yaml, (2) YAML parsing correctly converts to canonical domain types, (3) Caching mechanism prevents excessive network requests, (4) Error handling for network failures is robust, (5) Generated code reflects actual model capabilities. Rate 1-10 and provide specific feedback. Any score below 9 requires rework.

### 9. Fix YamlProcessor ModelInfo Compatibility ⭐ MEDIUM
- **File**: `packages/provider/build/yaml_processor.rs:42-80`
- **Issue**: YamlProcessor defines its own ModelInfo struct instead of using canonical domain version
- **Architecture**: YamlProcessor should parse external YAML and convert to canonical fluent_ai_domain::model::ModelInfo
- **Implementation**:
  - Remove local ModelInfo struct definition (lines 42-80)
  - Import fluent_ai_domain::model::ModelInfo as the canonical type
  - Update parsing methods to convert external YAML format to canonical ModelInfo structure
  - Ensure field mapping handles differences between external YAML and canonical structure
  - Update validation methods to work with canonical ModelInfo fields
  - Fix test cases to use canonical ModelInfo structure
- **Files to Modify**:
  - `packages/provider/build/yaml_processor.rs` (lines 42-80) - Remove duplicate ModelInfo
  - `packages/provider/build/yaml_processor.rs` (lines 150-200) - Update parsing methods
  - `packages/provider/build/yaml_processor.rs` (lines 280-319) - Fix test cases
- **Priority**: MEDIUM - Required for proper type compatibility
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 10. QA: YamlProcessor ModelInfo Compatibility Quality Assessment
Act as an Objective QA Rust developer and rate the work performed previously on fixing YamlProcessor ModelInfo compatibility. Evaluate: (1) Proper removal of duplicate ModelInfo struct, (2) Correct import and usage of canonical domain ModelInfo, (3) Field mapping handles external YAML format correctly, (4) Validation methods work with canonical structure, (5) Test cases updated properly. Rate 1-10 and provide specific feedback. Any score below 9 requires rework.

### 11. Add Missing Build Dependencies ⭐ MEDIUM
- **File**: `packages/provider/Cargo.toml` (build-dependencies section)
- **Issue**: Build script likely needs additional dependencies for HTTP requests, YAML parsing, and template processing
- **Architecture**: Add necessary build dependencies without affecting runtime dependencies
- **Implementation**:
  - Add `reqwest` with blocking feature for HTTP requests in build script
  - Add `serde_yaml` for YAML parsing in build script
  - Add `tera` or `handlebars` for template processing
  - Add `tokio` with minimal features for async HTTP in build script
  - Ensure all dependencies are in [build-dependencies] section only
  - Use latest stable versions via `cargo search` for each dependency
- **Files to Modify**:
  - `packages/provider/Cargo.toml` - Add build-dependencies section with required crates
- **Priority**: MEDIUM - Required for build script functionality
- **Command**: Use `cargo add --build <dependency>` to add build dependencies properly
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 12. QA: Build Dependencies Quality Assessment
Act as an Objective QA Rust developer and rate the work performed previously on adding missing build dependencies. Evaluate: (1) All required dependencies added to [build-dependencies] section, (2) Latest stable versions used, (3) Minimal feature sets selected to avoid bloat, (4) Dependencies support the required functionality (HTTP, YAML, templates), (5) No conflicts with existing dependencies. Rate 1-10 and provide specific feedback. Any score below 9 requires rework.

### 13. Fix Architecture-Specific Macro Issues ⭐ MEDIUM
- **File**: `packages/provider/src/clients/openrouter/streaming.rs:1333-1335`
- **Issue**: x86/x86_64 specific macros used on ARM architecture causing compilation failures
- **Architecture**: Add proper architecture guards or use cross-platform alternatives
- **Implementation**:
  - Wrap architecture-specific macros with `cfg(any(target_arch = "x86", target_arch = "x86_64"))` guards
  - Add ARM64 alternatives with `cfg(target_arch = "aarch64")` guards
  - Use cross-platform alternatives where possible instead of architecture-specific code
  - Ensure functionality works correctly on all supported architectures
  - Add compile-time feature detection for SIMD capabilities
- **Files to Modify**:
  - `packages/provider/src/clients/openrouter/streaming.rs` (lines 1333-1335) - Add architecture guards
- **Priority**: MEDIUM - Fixes compilation on ARM architectures
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 14. QA: Architecture-Specific Macro Fixes Quality Assessment
Act as an Objective QA Rust developer and rate the work performed previously on fixing architecture-specific macro issues. Evaluate: (1) Proper cfg guards for x86/x86_64 architectures, (2) ARM64 alternatives provided where needed, (3) Cross-platform compatibility maintained, (4) SIMD feature detection works correctly, (5) Code compiles on all target architectures. Rate 1-10 and provide specific feedback. Any score below 9 requires rework.

### 15. Add Missing Candle Package Dependencies ⭐ LOW
- **File**: `packages/fluent-ai-candle/Cargo.toml`
- **Issue**: Missing `fastrand` and `rand` dependencies causing compilation failures
- **Architecture**: Add required dependencies for random number generation in Candle package
- **Implementation**:
  - Add `fastrand` dependency with latest stable version
  - Add `rand` dependency with latest stable version
  - Ensure versions are compatible with existing Candle dependencies
  - Use `cargo search` to find latest stable versions
- **Files to Modify**:
  - `packages/fluent-ai-candle/Cargo.toml` - Add missing dependencies
- **Priority**: LOW - Fixes specific package compilation issues
- **Command**: Use `cargo add fastrand rand` in fluent-ai-candle directory
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 16. QA: Candle Package Dependencies Quality Assessment
Act as an Objective QA Rust developer and rate the work performed previously on adding missing Candle package dependencies. Evaluate: (1) Correct dependencies added (fastrand, rand), (2) Latest stable versions used, (3) Version compatibility with existing Candle dependencies, (4) Dependencies resolve compilation errors, (5) No unnecessary dependencies added. Rate 1-10 and provide specific feedback. Any score below 9 requires rework.

## SYSTEMATIC IMPLEMENTATION APPROACH

### Phase 1: Core Build Script Reconstruction (Items 1-2)
- Replace stub build.rs with full code generation system
- Integrate with existing CodeGenerator and YamlProcessor infrastructure
- **Success Criteria**: Build script runs and attempts code generation

### Phase 2: Code Generation Infrastructure (Items 3-6)
- Restore missing CodeGenerator methods
- Fix template compatibility with canonical domain types
- **Success Criteria**: Templates generate syntactically correct Rust code

### Phase 3: External Data Integration (Items 7-10)
- Integrate external model data source (sigoden's models.yaml)
- Fix YamlProcessor compatibility with canonical types
- **Success Criteria**: Generated code uses real model data and canonical types

### Phase 4: Dependency and Platform Fixes (Items 11-16)
- Add missing build dependencies
- Fix architecture-specific macro issues
- Add missing Candle package dependencies
- **Success Criteria**: All packages compile successfully across platforms

## FINAL VALIDATION

### Comprehensive Quality Assurance
- Run `cargo check` on entire workspace to verify 0 errors and 0 warnings
- Verify generated provider and model code exists and is properly structured
- Test compilation on multiple architectures (x86_64, ARM64)
- Validate that all generated code uses canonical domain types
- Confirm external model data is properly integrated and cached

### Success Metrics
- **BEFORE**: 600+ compilation errors across workspace
- **AFTER**: 0 errors, 0 warnings across entire workspace
- **Generated Code**: Proper provider enums, model definitions, client implementations
- **Type Safety**: All code uses canonical fluent_ai_domain::model::ModelInfo
- **Platform Support**: Compiles successfully on all target architectures

---

# CONSTRAINTS AND REQUIREMENTS

## Code Quality Standards
- Never use `unwrap()` in src/* or examples/*
- Never use `expect()` in src/* or examples/*
- DO USE `expect()` in ./tests/* only
- DO NOT use `unwrap()` in ./tests/*
- All generated code must follow these standards

## Implementation Standards
- Make ONLY THE MINIMAL, SURGICAL CHANGES required
- Do not modify or rewrite any portion of the app outside scope
- Use production-quality, zero-allocation patterns where possible
- Maintain backward compatibility with existing APIs
- Follow existing code style and architecture patterns

## Dependency Management
- Use `cargo add` commands for adding dependencies
- Never directly edit Cargo.toml files
- Use latest stable versions via `cargo search`
- Minimize dependency footprint and feature sets