# Unused Imports Cleanup

## Description
Remove all unused imports (186 warnings) to achieve zero-warning compilation. This cleanup task can be performed after core compilation issues are resolved.

## Success Criteria
- All 186 unused import warnings eliminated
- No functionality broken by import removal
- Clean, minimal import statements
- Zero compilation warnings

## Dependencies
- Must complete after: All tasks in 0_core_module_fixes/ and 1_independent_fixes/

## Estimated Complexity
**Low** - Mechanical cleanup task

## Technical Details

### Categories of Unused Imports

#### TLS Module Cleanup
- `tls/certificate.rs:5` - Remove unused `BufRead`
- `tls/mod.rs:59` - Remove unused TLS connector imports
- `tls/mod.rs:64` - Remove unused `Cert`, `ClientCert`

#### WASM Module Cleanup  
- `wasm/body/` - Multiple unused type imports
- `wasm/client/` - Extensive unused HTTP type imports
- `wasm/request/` - Unused builder and type imports
- `wasm/response.rs` - Unused error handling imports

#### JSON Path Module Cleanup
- `json_path/buffer/` - Unused buffer type imports
- `json_path/error/` - Unused error constructor imports  
- `json_path/stream_processor/` - Unused processing imports
- `json_path/selector_parser/` - Unused parser imports

### Implementation Strategy
1. **Automated Cleanup**: Use `cargo clippy --fix` for safe removals
2. **Manual Review**: Check each import for hidden usage
3. **Batch Processing**: Group by module for efficient cleanup
4. **Validation**: Ensure no functionality regression

### Commands to Execute
```bash
# Run clippy to identify and fix unused imports
cargo clippy --fix --allow-dirty --allow-staged

# Manual verification
cargo check --message-format short --quiet
```

### Files Affected
- 50+ files across tls/, wasm/, json_path/ modules
- See TODO.md lines 140-188 for complete list

### Validation Steps
1. Run `cargo check --message-format short --quiet` - should show 0 warnings
2. Run full test suite to ensure no functionality broken
3. Verify all modules still compile correctly
4. Check that public API surface unchanged