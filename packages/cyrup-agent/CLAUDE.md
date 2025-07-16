# Architecture Overview

Direction update — **“Better RIG”** with zero-alloc hot-path + fluent async chains

Below is an API/architecture sketch that folds your new constraints into the earlier refactor roadmap.

## Guiding Invariants

### Core

ALL ASYNC WORK RETURNS AsyncStream of AsyncTrait from sync methods in public surfaces.

```rust
/// Zero-alloc bounded async stream (backed by crossbeam_ring)
pub struct AsyncStream<T, const CAP: usize> { /* non-blocking */ }

/// One-shot async task (crossbeam oneshot)
pub struct AsyncTask<T> { /* await -> Result<T,E> in matcher */ }
```

### Hot-path

- `crossbeam::ring::ArrayQueue<CAP>`: producer offers &'static mut T slot requiring no allocation once pre-filled.
- *Back-pressure*: AtomicUsize tracks queue length; on overflow we drop or coalesce (policy decided by sampler).

## Fluent chain anatomy (example: completion)

```rust
// ── provider side ─────────────────────────────────────────────
let reply_stream = CompletionProvider::openai()
    .model("o4-mini")
    .system_prompt("You are…") // -> AgentBuilder<MissingCtx>
    .context(doc_index.top_n())// -> AgentBuilder<Ready>
    .tool::<Calc>()            // const-generic tool registration
    .temperature(1.0)
    .completion() // builds CompletionProvider
    .on_chunk( | result | { // executed on each Stream chunk to unwrap
        Ok => result.into_chunk(),
        Err(e) => result.into_err!("agent failed: {e}")
    })
    .chat(
        "Hello! How's the new framework coming?"
    ); // returns unwrapped chunks in CompletionStream processed by on_chunk closure

// ── consumer side ─────────────────────────────────────────────
while let Some(reply) = reply_stream.next().await {
    println!("AI: {reply}");
}
```

### Key Points

- `openai()` static yields a polymorphic ProviderBuilder type-safe, typestate without new()
- `completion()` yields a Provider and invokes their CompletionBuilder
- `on_chunk()` takes a messaage and a closure Result (happy-path & error-path) to avoid unwrapping every CompletionStream chunk

## Non-blocking channels

Non-blocking channels & capacity negotiation:

```rust
/// zero-copy bounded ring (crossbeam)
static RING: ArrayQueue<Event<MAX_SIZE>> = ArrayQueue::new(MAX_SIZE);
static LEN: AtomicUsize = AtomicUsize::new(0);

// on produce
if LEN.fetch_add(1, AcqRel) < MAX_SIZE {
    RING.push(event).unwrap();
} else {
    LEN.fetch_sub(1, AcqRel);
    // policy: drop or block caller with exponential backoff
}
```

No Mutex; only atomics.

For dynamic workloads where MAX_SIZE not known at compile-time, expose const-generic selector:

```rust
/// zero-copy bounded ring (crossbeam)
static RING: ArrayQueue<Event<MAX_SIZE>> = ArrayQueue::new(MAX_SIZE);
static LEN: AtomicUsize = AtomicUsize::new(0);

// on produce
if LEN.fetch_add(1, AcqRel) < MAX_SIZE {
    RING.push(event).unwrap();
} else {
    LEN.fetch_sub(1, AcqRel);
    // policy: drop or block caller with exponential backoff
}
```

For dynamic workloads where MAX_SIZE not known at compile-time, expose const-generic selector:

```rust
type CompletionStream<T> = AsyncStream<T, { cfg::CHAT_CAPACITY }>;
```
