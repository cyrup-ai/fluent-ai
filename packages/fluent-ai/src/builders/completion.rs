//! Completion builder implementations
//!
//! All completion request construction logic and builder patterns.

use fluent_ai_domain::{AsyncTask, spawn_async, ZeroOneOrMany, Models};
use fluent_ai_domain::completion::{CompletionRequest, ToolDefinition};
use fluent_ai_domain::{Message, Document};
use serde_json::Value;

pub struct CompletionRequestBuilder {
    model: Option<Models>,
    system_prompt: Option<String>,
    chat_history: ZeroOneOrMany<Message>,
    documents: ZeroOneOrMany<Document>,
    tools: ZeroOneOrMany<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    chunk_size: Option<usize>,
    additional_params: Option<Value>,
}

pub struct CompletionRequestBuilderWithHandler {
    model: Option<Models>,
    system_prompt: Option<String>,
    chat_history: ZeroOneOrMany<Message>,
    documents: ZeroOneOrMany<Document>,
    tools: ZeroOneOrMany<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    chunk_size: Option<usize>,
    additional_params: Option<Value>,
    error_handler: Box<dyn Fn(String) + Send + Sync>,
    result_handler: Option<Box<dyn FnOnce(CompletionRequest) -> CompletionRequest + Send + 'static>>,
    chunk_handler: Option<Box<dyn FnMut(CompletionRequest) -> CompletionRequest + Send + 'static>>,
}

impl CompletionRequestBuilder {
    // Semantic entry point
    pub fn prompt(system_prompt: impl Into<String>) -> CompletionRequestBuilder {
        CompletionRequestBuilder {
            model: None,
            system_prompt: Some(system_prompt.into()),
            chat_history: ZeroOneOrMany::None,
            documents: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            temperature: None,
            max_tokens: None,
            chunk_size: None,
            additional_params: None,
        }
    }
    
    // New method instead of duplicating impl block
    pub fn model(mut self, model: Models) -> Self {
        self.model = Some(model);
        self
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn chat_history(mut self, history: ZeroOneOrMany<Message>) -> Self {
        self.chat_history = history;
        self
    }

    pub fn add_message(mut self, message: Message) -> Self {
        self.chat_history = match self.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, message]),
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::many(messages)
            },
        };
        self
    }

    pub fn user(mut self, content: impl Into<String>) -> Self {
        let message = Message::user(content);
        self.chat_history = match self.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, message]),
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::many(messages)
            },
        };
        self
    }

    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        let message = Message::assistant(content);
        self.chat_history = match self.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, message]),
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::many(messages)
            },
        };
        self
    }

    pub fn system(mut self, content: impl Into<String>) -> Self {
        let message = Message::system(content);
        self.chat_history = match self.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, message]),
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::many(messages)
            },
        };
        self
    }

    pub fn documents(mut self, documents: ZeroOneOrMany<Document>) -> Self {
        self.documents = documents;
        self
    }

    pub fn add_document(mut self, document: Document) -> Self {
        self.documents = match self.documents {
            ZeroOneOrMany::None => ZeroOneOrMany::One(document),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, document]),
            ZeroOneOrMany::Many(mut documents) => {
                documents.push(document);
                ZeroOneOrMany::many(documents)
            },
        };
        self
    }

    pub fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn add_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        let tool = ToolDefinition {
            name: name.into(),
            description: description.into(),
            parameters,
        };
        self.tools = match self.tools {
            ZeroOneOrMany::None => ZeroOneOrMany::One(tool),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, tool]),
            ZeroOneOrMany::Many(mut tools) => {
                tools.push(tool);
                ZeroOneOrMany::many(tools)
            },
        };
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, max: u64) -> Self {
        self.max_tokens = Some(max);
        self
    }

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    pub fn params<F>(mut self, f: F) -> Self
    where
        F: FnOnce() -> std::collections::HashMap<String, Value>,
    {
        let params = f();
        let json_params: serde_json::Map<String, Value> = params.into_iter().collect();
        self.additional_params = Some(Value::Object(json_params));
        self
    }

    // Error handling - required before terminal methods
    pub fn on_error<F>(self, handler: F) -> CompletionRequestBuilderWithHandler
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        CompletionRequestBuilderWithHandler {
            model: self.model,
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
            error_handler: Box::new(handler),
            result_handler: None,
            chunk_handler: None,
        }
    }

    pub fn on_result<F>(self, handler: F) -> CompletionRequestBuilderWithHandler
    where
        F: FnOnce(CompletionRequest) -> CompletionRequest + Send + 'static,
    {
        CompletionRequestBuilderWithHandler {
            model: self.model,
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
            error_handler: Box::new(|e| eprintln!("Completion error: {}", e)),
            result_handler: Some(Box::new(handler)),
            chunk_handler: None,
        }
    }

    pub fn on_chunk<F>(self, handler: F) -> CompletionRequestBuilderWithHandler
    where
        F: FnMut(CompletionRequest) -> CompletionRequest + Send + 'static,
    {
        CompletionRequestBuilderWithHandler {
            model: self.model,
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
            error_handler: Box::new(|e| eprintln!("Completion chunk error: {}", e)),
            result_handler: None,
            chunk_handler: Some(Box::new(handler)),
        }
    }
}

impl CompletionRequestBuilderWithHandler {
    // Terminal method - returns CompletionRequest
    pub fn request(self) -> CompletionRequest {
        CompletionRequest {
            system_prompt: self.system_prompt.unwrap_or_default(),
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
        }
    }

    // Terminal method - submits request and returns future
    pub fn complete<F>(self, _handler: F) -> AsyncTask<String>
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        spawn_async(async move {
            let request = CompletionRequest {
                system_prompt: self.system_prompt.unwrap_or_default(),
                chat_history: self.chat_history,
                documents: self.documents,
                tools: self.tools,
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                chunk_size: self.chunk_size,
                additional_params: self.additional_params,
            };

            // TODO: Implement actual completion logic
            format!("Completion for: {}", request.system_prompt)
        })
    }

    // Terminal method with result handling
    pub fn on_completion<F>(self, f: F) -> AsyncTask<String>
    where
        F: FnOnce(Result<String, String>) -> String + Send + 'static,
    {
        spawn_async(async move {
            let result = Ok("Completion response".to_string());
            f(result)
        })
    }
}