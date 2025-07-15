# InMemoryVectorStore

| Property | Type | Example |
|----------|------|---------|
| `embeddings` | `HashMap<String, (D, [OneOrMany](../one-or-many/)<[Embedding](../embedding/)>)>` | `HashMap::from([("doc1", (document1, OneOrMany::one(embedding1))), ("doc2", (document2, OneOrMany::many(vec![embedding2a, embedding2b])))])` |