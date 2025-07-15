# EmbeddingsBuilder

| Property | Type | Example |
|----------|------|---------|
| `model` | `M: EmbeddingModel` | `openai_client.embedding_model("text-embedding-ada-002")` |
| `documents` | `Vec<(T, Vec<String>)>` | `vec![(document1, vec!["text1", "text2"]), (document2, vec!["text3"])]` |