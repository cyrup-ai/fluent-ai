review /Volumes/samsung_t9/fluent-ai/packages/async-stream
review /Volumes/samsung_t9/fluent-ai/packages/http3

These set the "streams only" foundational building blocks for the whole library. ALL http calls MUST use http3 and ALL asynchronous work must use the Streams only, **No Futures**, all unwrapped Stream, no Result wrapped architecture.

 This "streams-first" architecture is what's unique and cool about fluent-ai, in addition to the [syntax "sugars"](./packages/fluent-ai/ARCHITECTURE.md) we provide, a [state of the art memory system](./packages/memory), an amazing, full featured, [daemon-based local MCP Server](./packages/sweetmcp/packages/daemon) with extensible [MCP tool plugins](./packages/sweetmcp/plugins), a [low allocation, robust simd implementation](./packages/simd), and a Rust-native [Candle LLM implementation](./packaes/candle) with first class support for the [very best huggingfaces models](/Volumes/samsung_t9/fluent-ai/packages/candle/src/providers/kimi_k2.rs).

Review samples of the ACTUAL REAL high level builders:

-/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/ARCHITECTURE.md
- /Volumes/samsung_t9/fluent-ai/packages/fluent-ai/examples/agent_role_builder.rs

^^
This is the kind of beautiful typesafe, fluent api we provide for end users of the library.

========================================

### [`model-info`](./packages/model-info)

ABOVE the root async-stream and http3 foundation, we have a [model-info package](./packages/model-info) that dynamically uses syn to build out strongly typed model configuration data, guaranteeing correctness and runtime/compile type safety with full enumerations using their respective /v1/models endpoints from the actual providers.

### [`domain`](./packages/domain)

All "value types' that are core and fundamental to an AI/LLM generation ecosystem are located here. The *BUILDERS Do NOT GO HERE* (they go in ./packages/fluent-ai, as does any other service logic). `domain` depends on `model-info` from which it obtains the Provider, Model and ModelInfo types that detail model capabilities.

### `termcolor`(./packages/termcolor)

This is a fork of termcolor that should be used for ALL output to the console for chat loops and other user facing dialog at runtime targeting the console/terminal.

### [`cylo`](./packages/cylo)

This library provides safe tool executions using platform specific containers and/or jail cells. It is deeply integrated into the tool execution flows.

### [`memory`]{./packages/memory)

This is a state of the art memory system that is self correcting, with both vector similarity and graph relationships using surrealdb (latest stable version 2.3.7). It also supports truly advanced [cognitive, quantum reasoning](/Volumes/samsung_t9/fluent-ai/packages/memory/src/cognitive). Don't mess with this unless you are specifically asked to do so.

### [`candle`](./packages/candle)

Candle project is special insomuch as it's designed to be totally stand-alone OR used in the fully integrated Fluent-AI environment. You may see what appears to be "duplicate" objects that are also in ./packages/domain and/or builder in ./packages/fluent-ai -- but this is intentional as we want it to be fully runnable as an standalone candle implementation for users who want that functionality but not the full (awesome) other features. [./packages/candle](./packages/candle) provides a first class, native Rust implementation of the [Candle LLM library](./tmp/candle). It supports all the latest and greatest models from HuggingFace, including the [Kimi-K2](https://huggingface.co/kimi-ai/kimi-k2) model which is the default model used in Fluent-AI.

=============================

## [CARGO HARKARI](./CARGO-HAKARI.md)

- NO DEPENDENCIES should ever use `workspace = true`. and NO dependency versions should be defined in ./Cargo.toml (at all). Instead, these are ALL defined in the specific packages and cargo hakari manages the rest.
- use `just hakari-regenerate` command which fully automates rebuilding dependencies and ALL updates after making dependency changes. DO NOT MESS with ./config/hakari.toml without asking David Maple for permission.
- ALWAYS use the VERY LATEST version of ALL DEPENDENCIES by checking with `cargo search` UNLESS David has specifically provided exception for a library.

## ASK QUESTIONS

This is a really cood architecture. If you don't know how to execute a task requested, ask questions. WE LOVE QUESTIONS.

- Do not ask questions without first researching with sequential thinking (your question should be informed by the state of the code that exists).
- Do offer choices if you are considering multiple approaches with pros/cons of each path.
- In general, I always 'want it done right', so if the question is "can we skip doing it right to make it work", my answer will always be "no".

## Recommend Improvements

- If you see something out of spec, let me know
- If you see something suboptimal, let me know
- If you have suggestions on better ergnomics or other infrastructure that MAKE YOU MORE PRODUCTIVE in this environment, let me know.

# `desktop-commander`

- use desktop commander for all cli command execution, file crud, searching and exporation of the code
- always run checks using `cargo check` after making changes to ensure they truly work BEFORE you give a "victory dance" summary of your work.

## [./TODO.md](./TODO.md)

Plan, track and disposition ALL YOUR WORK objectives, milestones and tasks in TODO.md. When editing it, DO NOT OVERWRITE THE WORK OR OTHER DEVS!

- First review it, identify if you are adding NEW work or if there are already related work items
  - if existing, MERGE your items in, improving the planning details from your research with specific details, citations to research sources (remote urls or local file paths) and detailed architectural notes.
  - if net new, CREATE them in a section clearly DEMARKATED from the other work in progress

After planning is approve:

- Create all work items with "STATUS: PLANNED"

As you work:

- mark each item in progress as "STATUS: WORKING"
- when finished, mark the items as "STATUS: QA READY"
- when qa is done, mark the item as "STATUS: READY FOR TESTING"
- when functional testing using the *real app* is done, mark the item as "STATUS: VERIFIED" and cross it off the list

** Periodically, I will ask you to remove from the TODO.md all verified status items to keep it focused and reduce noise **

====================================

## OBJECTIVE

Become a Fluent-AI EXPERT assistant

### FIRST TASK

Start by exploring the codebase packages/** with this background information and become a *"fluent-ai EXPERT"* ready to perform any task needed with a solid background. Then, ask me "David, what's my next task to work on?"

*use sequential thinking*
