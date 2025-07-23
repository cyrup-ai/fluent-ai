# TODO: Fix All Errors and Warnings to Zero 🎯

**Objective**: Achieve 0 errors and 0 warnings in the entire workspace with production-quality code.

## COMPILATION ERRORS (70 total)

### Lifetime & Borrow Checker Issues
1. [ ] Fix `borrowed data escapes outside of method` in `packages/domain/src/chat/config.rs:867`
2. [ ] QA: Rate fix quality (1-10) and provide feedback
3. [ ] Fix `borrowed data escapes outside of method` in `packages/domain/src/chat/search.rs:279`
4. [ ] QA: Rate fix quality (1-10) and provide feedback
5. [ ] Fix `borrowed data escapes outside of method` in `packages/domain/src/chat/search.rs:453`
6. [ ] QA: Rate fix quality (1-10) and provide feedback
7. [ ] Fix `borrowed data escapes outside of method` in `packages/domain/src/chat/search.rs:1221`
8. [ ] QA: Rate fix quality (1-10) and provide feedback
9. [ ] Fix `borrowed data escapes outside of method` in `packages/domain/src/chat/search.rs:1248`
10. [ ] QA: Rate fix quality (1-10) and provide feedback

### Type Mismatch & AsyncStream Issues
11. [ ] Fix missing argument in `AsyncStream::with_channel()` in `packages/domain/src/chat/commands/types.rs:608`
12. [ ] QA: Rate fix quality (1-10) and provide feedback
13. [ ] Fix type mismatch `expected (StreamingCommandExecutor, AsyncStream<CommandEvent>), found ()` in same location
14. [ ] QA: Rate fix quality (1-10) and provide feedback

### Moved Value Errors
15. [ ] Fix `use of moved value: builder` in `packages/domain/src/completion/request.rs:173`
16. [ ] QA: Rate fix quality (1-10) and provide feedback

### Additional Compilation Errors (67 more to be enumerated after initial fixes)
17. [ ] Enumerate and fix remaining 67 compilation errors systematically
18. [ ] QA: Rate fix quality (1-10) and provide feedback for each

## WARNINGS

### Missing Documentation (http3 package)
19. [ ] Add documentation for struct `BodyNotSet` in `packages/http3/src/builder.rs:284`
20. [ ] QA: Rate fix quality (1-10) and provide feedback
21. [ ] Add documentation for struct `BodySet` in `packages/http3/src/builder.rs:286`
22. [ ] QA: Rate fix quality (1-10) and provide feedback
23. [ ] Add documentation for trait `HttpStreamExt` in `packages/http3/src/builder.rs:289`
24. [ ] QA: Rate fix quality (1-10) and provide feedback
25. [ ] Add documentation for method `collect` in `packages/http3/src/builder.rs:290`
26. [ ] QA: Rate fix quality (1-10) and provide feedback
27. [ ] Add documentation for struct `HttpClientStats` in `packages/http3/src/client.rs:160`
28. [ ] QA: Rate fix quality (1-10) and provide feedback
29. [ ] Add documentation for all struct fields in `HttpClientStats` (8 fields)
30. [ ] QA: Rate fix quality (1-10) and provide feedback
31. [ ] Add documentation for struct `ClientStatsSnapshot` in `packages/http3/src/client.rs:173`
32. [ ] QA: Rate fix quality (1-10) and provide feedback
33. [ ] Add documentation for all struct fields in `ClientStatsSnapshot` (7 fields)
34. [ ] QA: Rate fix quality (1-10) and provide feedback

### Unused Code (Implement, Don't Remove)
35. [ ] Implement usage for field `start_time` in `packages/http3/src/client.rs:22` (HttpClient struct)
36. [ ] QA: Rate fix quality (1-10) and provide feedback
37. [ ] Implement usage for method `extract_request_cache_directives` in `packages/http3/src/middleware/cache.rs:80`
38. [ ] QA: Rate fix quality (1-10) and provide feedback
39. [ ] Implement usage for method `parse_max_age_directive` in `packages/http3/src/middleware/cache.rs:116`
40. [ ] QA: Rate fix quality (1-10) and provide feedback
41. [ ] Fix unused variables in domain package (multiple locations with `_variable` suggestions)
42. [ ] QA: Rate fix quality (1-10) and provide feedback

### Code Style Issues
43. [ ] Rename module `Header` to `header` (snake_case) in `packages/http3/src/builder.rs:48`
44. [ ] QA: Rate fix quality (1-10) and provide feedback

### Additional Warnings (to be enumerated)
45. [ ] Enumerate and fix all remaining warnings systematically
46. [ ] QA: Rate fix quality (1-10) and provide feedback for each

## VERIFICATION TASKS

47. [ ] Run `cargo check` after each fix to verify resolution
48. [ ] Ensure all dependencies are latest versions via `cargo search`
49. [ ] Test Kimi K2 example as end-user to verify functionality
50. [ ] Final verification: `cargo check` shows 0 errors and 0 warnings

## SUCCESS CRITERIA ✅

- [ ] **0 (Zero) compilation errors**
- [ ] **0 (Zero) warnings**
- [ ] **All code is production-quality**
- [ ] **Kimi K2 example runs successfully**
- [ ] **All QA ratings are 9 or 10**

---

**Notes:**
- Every warning is treated as a real code issue
- Unused code should be implemented, not removed (unless proven truly unnecessary after thorough review)
- Never cross off items without running `cargo check` to verify
- Never use blocking code without explicit permission
- Write zero-allocation, non-locking, async code
- Ask questions if unsure about any code modification