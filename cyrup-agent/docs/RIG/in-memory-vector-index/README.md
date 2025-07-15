# InMemoryVectorIndex

| Property | Type | Example |
|----------|------|---------|
| `model` | `M: EmbeddingModel` | `openai_client.embedding_model("text-embedding-ada-002")` |
| `store` | `[InMemoryVectorStore](../in-memory-vector-store/)<D>` | `InMemoryVectorStore::from_documents(documents)` |