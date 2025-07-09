use crate::domain::{tool::Tool, CompletionBackend};
use crate::AsyncTask;

pub struct AgentBuilder<B: CompletionBackend> {
    backend: B,
    system_prompt: Option<String>,
    tools: Vec<String>,
}

impl<B: CompletionBackend> AgentBuilder<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            system_prompt: None,
            tools: vec![],
        }
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn tool<T: Tool>(mut self, tool: T) -> Self {
        self.tools.push(tool.name().to_string());
        self
    }

    pub fn completion(self) -> CompletionProvider<B> {
        CompletionProvider {
            backend: self.backend,
            system_prompt: self.system_prompt.unwrap_or_default(),
            tools: self.tools,
        }
    }
}

pub struct CompletionProvider<B: CompletionBackend> {
    backend: B,
    system_prompt: String,
    tools: Vec<String>,
}

impl<B: CompletionBackend> CompletionProvider<B> {
    pub fn on_chunk<F>(self, f: F) -> ChunkHandler<B, F>
    where
        F: Fn(Result<String, String>) + Send + 'static,
    {
        ChunkHandler {
            backend: self.backend,
            system_prompt: self.system_prompt,
            tools: self.tools,
            handler: f,
        }
    }
}

pub struct ChunkHandler<B: CompletionBackend, F: Fn(Result<String, String>) + Send + 'static> {
    backend: B,
    system_prompt: String,
    tools: Vec<String>,
    handler: F,
}

impl<B, F> ChunkHandler<B, F>
where
    B: CompletionBackend + Send + 'static,
    F: Fn(Result<String, String>) + Send + 'static,
{
    pub fn chat(self, user_prompt: impl Into<String>) -> AsyncTask<()> {
        let prompt = format!("{} {}", self.system_prompt, user_prompt.into());
        let tools = self.tools;
        let handler = self.handler;

        AsyncTask::from_future(async move {
            match self.backend.submit_completion(&prompt, &tools).await {
                Ok(result) => handler(Ok(result)),
                Err(e) => handler(Err(e.to_string())),
            }
        })
    }
}
