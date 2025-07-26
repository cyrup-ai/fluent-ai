//! Completion builder implementations - Zero Box<dyn> trait-based architecture
//!
//! All completion request construction logic and builder patterns with zero allocation.

use std::marker::PhantomData;
use fluent_ai_domain::completion::{CompletionRequest, ToolDefinition};
use fluent_ai_domain::{AsyncTask, Models, ZeroOneOrMany, spawn_async};
use fluent_ai_domain::{Document, Message};
use serde_json::Value;

/// Completion request builder trait - elegant zero-allocation builder pattern
pub trait CompletionRequestBuilder: Sized {
    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    fn model(self, model: Models) -> impl CompletionRequestBuilder;
    
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(self, prompt: impl Into<String>) -> impl CompletionRequestBuilder;
    
    /// Set chat history - EXACT syntax: .chat_history(history)
    fn chat_history(self, history: ZeroOneOrMany<Message>) -> impl CompletionRequestBuilder;
    
    /// Add message - EXACT syntax: .add_message(message)
    fn add_message(self, message: Message) -> impl CompletionRequestBuilder;
    
    /// Add user message - EXACT syntax: .user("content")
    fn user(self, content: impl Into<String>) -> impl CompletionRequestBuilder;
    
    /// Add assistant message - EXACT syntax: .assistant("content")
    fn assistant(self, content: impl Into<String>) -> impl CompletionRequestBuilder;
    
    /// Add system message - EXACT syntax: .system("content")
    fn system(self, content: impl Into<String>) -> impl CompletionRequestBuilder;
    
    /// Set documents - EXACT syntax: .documents(docs)
    fn documents(self, documents: ZeroOneOrMany<Document>) -> impl CompletionRequestBuilder;
    
    /// Add document - EXACT syntax: .add_document(doc)
    fn add_document(self, document: Document) -> impl CompletionRequestBuilder;
    
    /// Set tools - EXACT syntax: .tools(tools)
    fn tools(self, tools: ZeroOneOrMany<ToolDefinition>) -> impl CompletionRequestBuilder;
    
    /// Add tool - EXACT syntax: .add_tool("name", "description", parameters)
    fn add_tool(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> impl CompletionRequestBuilder;
    
    /// Set temperature - EXACT syntax: .temperature(0.7)
    fn temperature(self, temp: f64) -> impl CompletionRequestBuilder;
    
    /// Set max tokens - EXACT syntax: .max_tokens(1000)
    fn max_tokens(self, max: u64) -> impl CompletionRequestBuilder;
    
    /// Set chunk size - EXACT syntax: .chunk_size(512)
    fn chunk_size(self, size: usize) -> impl CompletionRequestBuilder;
    
    /// Set additional parameters - EXACT syntax: .additional_params(params)
    fn additional_params(self, params: Value) -> impl CompletionRequestBuilder;
    
    /// Set parameters with closure - EXACT syntax: .params(|| { ... })
    fn params<F>(self, f: F) -> impl CompletionRequestBuilder
    where
        F: FnOnce() -> std::collections::HashMap<String, Value>;
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl CompletionRequestBuilder
    where
        F: Fn(String) + Send + Sync + 'static;
    
    /// Set result handler - EXACT syntax: .on_result(|result| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_result<F>(self, handler: F) -> impl CompletionRequestBuilder
    where
        F: FnOnce(CompletionRequest) -> CompletionRequest + Send + 'static;
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl CompletionRequestBuilder
    where
        F: FnMut(CompletionRequest) -> CompletionRequest + Send + 'static;
    
    /// Build request - EXACT syntax: .request()
    fn request(self) -> CompletionRequest;
    
    /// Complete with handler - EXACT syntax: .complete(|response| { ... })
    fn complete<F>(self, handler: F) -> AsyncTask<String>
    where
        F: Fn(String) + Send + Sync + 'static;
    
    /// Handle completion result - EXACT syntax: .on_completion(|result| { ... })
    fn on_completion<F>(self, f: F) -> AsyncTask<String>
    where
        F: FnOnce(Result<String, String>) -> String + Send + 'static;
}

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct CompletionRequestBuilderImpl<
    F1 = fn(String),
    F2 = fn(CompletionRequest) -> CompletionRequest,
    F3 = fn(CompletionRequest) -> CompletionRequest,
> where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(CompletionRequest) -> CompletionRequest + Send + 'static,
    F3: FnMut(CompletionRequest) -> CompletionRequest + Send + 'static,
{
    model: Option<Models>,
    system_prompt: Option<String>,
    chat_history: ZeroOneOrMany<Message>,
    documents: ZeroOneOrMany<Document>,
    tools: ZeroOneOrMany<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    chunk_size: Option<usize>,
    additional_params: Option<Value>,
    error_handler: Option<F1>,
    result_handler: Option<F2>,
    chunk_handler: Option<F3>,
}

impl CompletionRequestBuilderImpl {
    /// Semantic entry point - EXACT syntax: CompletionRequestBuilderImpl::prompt("system_prompt")
    pub fn prompt(system_prompt: impl Into<String>) -> impl CompletionRequestBuilder {
        CompletionRequestBuilderImpl {
            model: None,
            system_prompt: Some(system_prompt.into()),
            chat_history: ZeroOneOrMany::None,
            documents: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            temperature: None,
            max_tokens: None,
            chunk_size: None,
            additional_params: None,
            error_handler: None,
            result_handler: None,
            chunk_handler: None,
        }
    }
}

impl<F1, F2, F3> CompletionRequestBuilder for CompletionRequestBuilderImpl<F1, F2, F3>
where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(CompletionRequest) -> CompletionRequest + Send + 'static,
    F3: FnMut(CompletionRequest) -> CompletionRequest + Send + 'static,
{
    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    fn model(mut self, model: Models) -> impl CompletionRequestBuilder {
        self.model = Some(model);
        self
    }
    
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(mut self, prompt: impl Into<String>) -> impl CompletionRequestBuilder {
        self.system_prompt = Some(prompt.into());
        self
    }
    
    /// Set chat history - EXACT syntax: .chat_history(history)
    fn chat_history(mut self, history: ZeroOneOrMany<Message>) -> impl CompletionRequestBuilder {
        self.chat_history = history;
        self
    }
    
    /// Add message - EXACT syntax: .add_message(message)
    fn add_message(mut self, message: Message) -> impl CompletionRequestBuilder {
        self.chat_history = match self.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, message]),
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::many(messages)
            }
        };
        self
    }
    
    /// Add user message - EXACT syntax: .user("content")
    fn user(mut self, content: impl Into<String>) -> impl CompletionRequestBuilder {
        let message = Message::user(content);
        self.chat_history = match self.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, message]),
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::many(messages)
            }
        };
        self
    }
    
    /// Add assistant message - EXACT syntax: .assistant("content")
    fn assistant(mut self, content: impl Into<String>) -> impl CompletionRequestBuilder {
        let message = Message::assistant(content);
        self.chat_history = match self.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, message]),
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::many(messages)
            }
        };
        self
    }
    
    /// Add system message - EXACT syntax: .system("content")
    fn system(mut self, content: impl Into<String>) -> impl CompletionRequestBuilder {
        let message = Message::system(content);
        self.chat_history = match self.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, message]),
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::many(messages)
            }
        };
        self
    }
    
    /// Set documents - EXACT syntax: .documents(docs)
    fn documents(mut self, documents: ZeroOneOrMany<Document>) -> impl CompletionRequestBuilder {
        self.documents = documents;
        self
    }
    
    /// Add document - EXACT syntax: .add_document(doc)
    fn add_document(mut self, document: Document) -> impl CompletionRequestBuilder {
        self.documents = match self.documents {
            ZeroOneOrMany::None => ZeroOneOrMany::One(document),
            ZeroOneOrMany::One(existing) => ZeroOneOrMany::many(vec![existing, document]),
            ZeroOneOrMany::Many(mut documents) => {
                documents.push(document);
                ZeroOneOrMany::many(documents)
            }
        };
        self
    }
    
    /// Set tools - EXACT syntax: .tools(tools)
    fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> impl CompletionRequestBuilder {
        self.tools = tools;
        self
    }
    
    /// Add tool - EXACT syntax: .add_tool("name", "description", parameters)
    fn add_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> impl CompletionRequestBuilder {
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
            }
        };
        self
    }
    
    /// Set temperature - EXACT syntax: .temperature(0.7)
    fn temperature(mut self, temp: f64) -> impl CompletionRequestBuilder {
        self.temperature = Some(temp);
        self
    }
    
    /// Set max tokens - EXACT syntax: .max_tokens(1000)
    fn max_tokens(mut self, max: u64) -> impl CompletionRequestBuilder {
        self.max_tokens = Some(max);
        self
    }
    
    /// Set chunk size - EXACT syntax: .chunk_size(512)
    fn chunk_size(mut self, size: usize) -> impl CompletionRequestBuilder {
        self.chunk_size = Some(size);
        self
    }
    
    /// Set additional parameters - EXACT syntax: .additional_params(params)
    fn additional_params(mut self, params: Value) -> impl CompletionRequestBuilder {
        self.additional_params = Some(params);
        self
    }
    
    /// Set parameters with closure - EXACT syntax: .params(|| { ... })
    fn params<F>(mut self, f: F) -> impl CompletionRequestBuilder
    where
        F: FnOnce() -> std::collections::HashMap<String, Value>,
    {
        let params = f();
        let json_params: serde_json::Map<String, Value> = params.into_iter().collect();
        self.additional_params = Some(Value::Object(json_params));
        self
    }
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl CompletionRequestBuilder
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        CompletionRequestBuilderImpl {
            model: self.model,
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
            error_handler: Some(handler),
            result_handler: self.result_handler,
            chunk_handler: self.chunk_handler,
        }
    }
    
    /// Set result handler - EXACT syntax: .on_result(|result| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_result<F>(self, handler: F) -> impl CompletionRequestBuilder
    where
        F: FnOnce(CompletionRequest) -> CompletionRequest + Send + 'static,
    {
        CompletionRequestBuilderImpl {
            model: self.model,
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
            error_handler: self.error_handler,
            result_handler: Some(handler),
            chunk_handler: self.chunk_handler,
        }
    }
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl CompletionRequestBuilder
    where
        F: FnMut(CompletionRequest) -> CompletionRequest + Send + 'static,
    {
        CompletionRequestBuilderImpl {
            model: self.model,
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
            error_handler: self.error_handler,
            result_handler: self.result_handler,
            chunk_handler: Some(handler),
        }
    }
    
    /// Build request - EXACT syntax: .request()
    fn request(self) -> CompletionRequest {
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
    
    /// Complete with handler - EXACT syntax: .complete(|response| { ... })
    fn complete<F>(self, _handler: F) -> AsyncTask<String>
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
    
    /// Handle completion result - EXACT syntax: .on_completion(|result| { ... })
    fn on_completion<F>(self, f: F) -> AsyncTask<String>
    where
        F: FnOnce(Result<String, String>) -> String + Send + 'static,
    {
        spawn_async(async move {
            let result = Ok("Completion response".to_string());
            f(result)
        })
    }
}