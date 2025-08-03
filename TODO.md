# TODO

## HIGH PRIORITY TASKS

**✅ RFC COMPLIANCE COMPLETE**
- ✅ **COMPLETED** Add XPath equivalence tests (RFC Appendix B) - XPath to JSONPath mapping validation (Table 21)

## General Issues
- Fix model-info build script API key handling (missing environment fallbacks causing "No models found for openai" error)
- Resolve 88+ compilation warnings (missing documentation, unused imports, dead code)

## JSONPath Feature Completion  
- Implement buffer shrinking optimization in json_path/buffer.rs (BytesMut limitation workaround)
- Complete HTTP3 Builder integration with existing JSONPath functionality
- Wire JSONPath processing into HTTP response handling
- Implement JsonStreamProcessor for HTTP response chunk handling  
- Add JsonArrayStream integration with Http3Builder

---

## RFC 9535 COMPLIANCE STATUS ✅

**COMPREHENSIVE TEST COVERAGE EXISTS FOR:**
- ✅ ABNF Grammar (Appendix A) → rfc9535_abnf_compliance.rs
- ✅ Official Examples 13/13 → rfc9535_official_examples.rs ✅ **COMPLETE**
- ✅ Root Identifier (2.2) → rfc9535_abnf_compliance.rs + rfc9535_core_syntax.rs  
- ✅ Selectors 2.3.1-2.3.5 → rfc9535_selectors.rs + related files
- ✅ Function Extensions 2.4.4-2.4.8 → rfc9535_function_*.rs files (length, count, match, search, value)
- ✅ Function Type System 2.4.1-2.4.3 → rfc9535_function_type_system.rs + rfc9535_function_well_typedness.rs
- ✅ Segments 2.5.1-2.5.2 → rfc9535_segments.rs + rfc9535_segment_traversal.rs
- ✅ Null Semantics 2.6 → rfc9535_null_semantics.rs
- ✅ Normalized Paths 2.7 → rfc9535_normalized_paths.rs
- ✅ Security Considerations (Section 4) → rfc9535_security_compliance.rs
- ✅ Unicode Compliance → rfc9535_unicode_compliance.rs
- ✅ I-JSON Compliance → rfc9535_abnf_compliance.rs + rfc9535_comparison_edge_cases.rs
- ✅ Array Slice Algorithms → rfc9535_array_slice_*.rs files
- ✅ Filter Logic → rfc9535_filter_*.rs files
- ✅ Performance/Streaming → rfc9535_performance_compliance.rs + rfc9535_streaming_behavior.rs
- ✅ Error Handling → rfc9535_error_handling.rs
- ✅ String Literals → rfc9535_string_literals.rs
- ✅ Singular Queries → rfc9535_singular_queries.rs

- ✅ XPath Equivalence (Appendix B) → rfc9535_xpath_equivalence.rs

**Total: 28 RFC test files with comprehensive coverage - 🎉 100% RFC 9535 COMPLIANCE ACHIEVED 🎉**

**RFC 9535 Official Examples Status:**
- ✅ **COMPLETE: 13/13 official examples now covered** (including newly added: `$..book[2].author`, `$..book[2].publisher`, `$..*`)