use futures::stream::Stream;
use futures::StreamExt;

use crate::domain::chunk::CompletionChunk;
use crate::markdown::{MarkdownRenderer, MarkdownTheme, render_markdown_to_string};
use crate::async_task::AsyncStream;

/// Zero-allocation streaming output handler
pub struct StreamingOutput {
    markdown_renderer: Option<MarkdownRenderer>,
    enable_colors: bool,
    buffer: String,
}

impl StreamingOutput {
    /// Create new streaming output handler
    pub fn new() -> Self {
        Self {
            markdown_renderer: Some(MarkdownRenderer::new(MarkdownTheme::default())),
            enable_colors: true,
            buffer: String::with_capacity(4096),
        }
    }

    /// Create streaming output with custom theme
    pub fn with_theme(theme: MarkdownTheme) -> Self {
        Self {
            markdown_renderer: Some(MarkdownRenderer::new(theme)),
            enable_colors: true,
            buffer: String::with_capacity(4096),
        }
    }

    /// Create streaming output without markdown rendering
    pub fn plain() -> Self {
        Self {
            markdown_renderer: None,
            enable_colors: false,
            buffer: String::with_capacity(4096),
        }
    }

    /// Enable or disable color output
    pub fn with_colors(mut self, enable: bool) -> Self {
        self.enable_colors = enable;
        self
    }

    /// Process and output a chunk of text
    pub fn output_chunk(&mut self, chunk: &str) {
        if self.enable_colors && self.markdown_renderer.is_some() {
            // Use markdown rendering for rich output
            let rendered = render_markdown_to_string(chunk);
            print!("{}", rendered);
        } else {
            // Plain text output
            print!("{}", chunk);
        }

        // Flush immediately for real-time streaming
        use std::io::{self, Write};
        let _ = io::stdout().flush();
    }

    /// Process and output a complete message
    pub fn output_message(&mut self, message: &str) {
        self.buffer.clear();
        self.buffer.push_str(message);

        if self.enable_colors && self.markdown_renderer.is_some() {
            let rendered = render_markdown_to_string(&self.buffer);
            println!("{}", rendered);
        } else {
            println!("{}", message);
        }
    }

    /// Start a new line
    pub fn newline(&self) {
        println!();
    }

    /// Clear the current line (terminal control)
    pub fn clear_line(&self) {
        if self.enable_colors {
            print!("\r\x1b[K");
            use std::io::{self, Write};
            let _ = io::stdout().flush();
        }
    }
}

impl Default for StreamingOutput {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for streaming content processors
pub trait StreamProcessor {
    type Item;
    type Error;

    /// Process a single streaming item
    fn process_item(&mut self, item: Self::Item) -> Result<(), Self::Error>;

    /// Handle end of stream
    fn finish(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Advanced completion chunk streaming processor
pub struct CompletionStreamProcessor {
    output: StreamingOutput,
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    current_tool_input: String,
    accumulated_text: String,
    show_tool_calls: bool,
    show_usage_stats: bool,
}

impl CompletionStreamProcessor {
    pub fn new() -> Self {
        Self {
            output: StreamingOutput::new(),
            current_tool_id: None,
            current_tool_name: None,
            current_tool_input: String::new(),
            accumulated_text: String::new(),
            show_tool_calls: true,
            show_usage_stats: true,
        }
    }

    pub fn with_output(output: StreamingOutput) -> Self {
        Self {
            output,
            current_tool_id: None,
            current_tool_name: None,
            current_tool_input: String::new(),
            accumulated_text: String::new(),
            show_tool_calls: true,
            show_usage_stats: true,
        }
    }

    pub fn show_tool_calls(mut self, show: bool) -> Self {
        self.show_tool_calls = show;
        self
    }

    pub fn show_usage_stats(mut self, show: bool) -> Self {
        self.show_usage_stats = show;
        self
    }

    fn format_tool_call_start(&self, id: &str, name: &str) -> String {
        if self.show_tool_calls {
            format!("\n🔧 **Tool Call**: {} ({})\n", name, &id[..8])
        } else {
            String::new()
        }
    }

    fn format_tool_call_complete(&self, name: &str, input: &str) -> String {
        if self.show_tool_calls {
            format!("📋 **Tool Input**: {}\n✅ **Tool Complete**: {}\n", 
                   input.chars().take(100).collect::<String>(), name)
        } else {
            String::new()
        }
    }

    fn format_usage_stats(&self, usage: &crate::domain::chunk::Usage) -> String {
        if self.show_usage_stats {
            format!("\n📊 **Usage**: {} input, {} output, {} total tokens\n",
                   usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        } else {
            String::new()
        }
    }
}

impl StreamProcessor for CompletionStreamProcessor {
    type Item = CompletionChunk;
    type Error = std::io::Error;

    fn process_item(&mut self, item: Self::Item) -> Result<(), Self::Error> {
        match item {
            CompletionChunk::Text(text) => {
                self.accumulated_text.push_str(&text);
                self.output.output_chunk(&text);
            }
            CompletionChunk::ToolCallStart { id, name } => {
                let formatted = self.format_tool_call_start(&id, &name);
                if !formatted.is_empty() {
                    self.output.output_chunk(&formatted);
                }
                self.current_tool_id = Some(id);
                self.current_tool_name = Some(name);
                self.current_tool_input.clear();
            }
            CompletionChunk::ToolCall { partial_input, .. } => {
                self.current_tool_input.push_str(&partial_input);
            }
            CompletionChunk::ToolCallComplete { name, input, .. } => {
                let formatted = self.format_tool_call_complete(&name, &input);
                if !formatted.is_empty() {
                    self.output.output_chunk(&formatted);
                }
                self.current_tool_id = None;
                self.current_tool_name = None;
                self.current_tool_input.clear();
            }
            CompletionChunk::Complete { usage, .. } => {
                if let Some(usage) = usage {
                    let formatted = self.format_usage_stats(&usage);
                    if !formatted.is_empty() {
                        self.output.output_chunk(&formatted);
                    }
                }
            }
            CompletionChunk::Error(error) => {
                let formatted = format!("\n❌ **Error**: {}\n", error);
                self.output.output_chunk(&formatted);
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Self::Error> {
        self.output.newline();
        Ok(())
    }
}

/// Simple text streaming processor
pub struct TextStreamProcessor {
    output: StreamingOutput,
}

impl TextStreamProcessor {
    pub fn new() -> Self {
        Self {
            output: StreamingOutput::new(),
        }
    }

    pub fn with_output(output: StreamingOutput) -> Self {
        Self { output }
    }
}

impl Default for TextStreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamProcessor for TextStreamProcessor {
    type Item = String;
    type Error = std::io::Error;

    fn process_item(&mut self, item: Self::Item) -> Result<(), Self::Error> {
        self.output.output_chunk(&item);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Self::Error> {
        self.output.newline();
        Ok(())
    }
}

/// Adapter to make any Stream compatible with StreamProcessor
pub struct StreamAdapter<S, P>
where
    S: Stream,
    P: StreamProcessor,
{
    stream: S,
    processor: P,
}

impl<S, P> StreamAdapter<S, P>
where
    S: Stream,
    P: StreamProcessor,
{
    pub fn new(stream: S, processor: P) -> Self {
        Self { stream, processor }
    }

    /// Process the entire stream
    pub async fn process_all(mut self) -> Result<(), P::Error>
    where
        S: Stream<Item = P::Item> + Unpin,
    {
        use futures::StreamExt;

        while let Some(item) = self.stream.next().await {
            self.processor.process_item(item)?;
        }

        self.processor.finish()
    }
}

/// High-performance streaming aggregator for completion chunks
pub struct StreamingAggregator {
    accumulated_text: String,
    tool_calls: Vec<ToolCallInfo>,
    usage_stats: Option<crate::domain::chunk::Usage>,
    error: Option<String>,
    is_complete: bool,
}

#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub input: String,
    pub is_complete: bool,
}

impl StreamingAggregator {
    /// Create new streaming aggregator
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            accumulated_text: String::with_capacity(4096),
            tool_calls: Vec::new(),
            usage_stats: None,
            error: None,
            is_complete: false,
        }
    }

    /// Process a completion chunk and update internal state
    #[inline(always)]
    pub fn process_chunk(&mut self, chunk: CompletionChunk) {
        match chunk {
            CompletionChunk::Text(text) => {
                self.accumulated_text.push_str(&text);
            }
            CompletionChunk::ToolCallStart { id, name } => {
                self.tool_calls.push(ToolCallInfo {
                    id,
                    name,
                    input: String::new(),
                    is_complete: false,
                });
            }
            CompletionChunk::ToolCall { partial_input, .. } => {
                if let Some(call) = self.tool_calls.last_mut() {
                    call.input.push_str(&partial_input);
                }
            }
            CompletionChunk::ToolCallComplete { id, input, .. } => {
                if let Some(call) = self.tool_calls.iter_mut().find(|c| c.id == id) {
                    call.input = input;
                    call.is_complete = true;
                }
            }
            CompletionChunk::Complete { usage, .. } => {
                self.usage_stats = usage;
                self.is_complete = true;
            }
            CompletionChunk::Error(error) => {
                self.error = Some(error);
            }
        }
    }

    /// Get accumulated text content
    #[inline(always)]
    pub fn text(&self) -> &str {
        &self.accumulated_text
    }

    /// Get tool calls
    #[inline(always)]
    pub fn tool_calls(&self) -> &[ToolCallInfo] {
        &self.tool_calls
    }

    /// Get usage statistics
    #[inline(always)]
    pub fn usage(&self) -> Option<&crate::domain::chunk::Usage> {
        self.usage_stats.as_ref()
    }

    /// Check if completion is finished
    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Get error if any occurred
    #[inline(always)]
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Process entire AsyncStream and return final aggregated result
    pub async fn process_stream(mut stream: AsyncStream<CompletionChunk>) -> Self {
        let mut aggregator = Self::new();
        
        while let Some(chunk) = stream.next().await {
            aggregator.process_chunk(chunk);
            if aggregator.is_complete {
                break;
            }
        }
        
        aggregator
    }
}

impl Default for StreamingAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for streaming
pub mod utils {
    use super::*;

    /// Create a streaming output handler optimized for chat
    pub fn chat_output() -> StreamingOutput {
        StreamingOutput::new().with_colors(true)
    }

    /// Create a plain text streaming output handler
    pub fn plain_output() -> StreamingOutput {
        StreamingOutput::plain()
    }

    /// Create a completion stream processor with tool call support
    pub fn completion_processor() -> CompletionStreamProcessor {
        CompletionStreamProcessor::new()
    }

    /// Create a completion stream processor without tool call display
    pub fn completion_processor_text_only() -> CompletionStreamProcessor {
        CompletionStreamProcessor::new()
            .show_tool_calls(false)
            .show_usage_stats(false)
    }

    /// Process a text stream with markdown rendering
    pub async fn process_text_stream<S>(stream: S) -> Result<(), std::io::Error>
    where
        S: Stream<Item = String> + Unpin,
    {
        let processor = TextStreamProcessor::new();
        let adapter = StreamAdapter::new(stream, processor);
        adapter.process_all().await
    }

    /// Process a text stream with custom output handler
    pub async fn process_text_stream_with_output<S>(
        stream: S,
        output: StreamingOutput,
    ) -> Result<(), std::io::Error>
    where
        S: Stream<Item = String> + Unpin,
    {
        let processor = TextStreamProcessor::with_output(output);
        let adapter = StreamAdapter::new(stream, processor);
        adapter.process_all().await
    }

    /// Process a completion chunk stream with real-time output
    pub async fn process_completion_stream<S>(stream: S) -> Result<(), std::io::Error>
    where
        S: Stream<Item = CompletionChunk> + Unpin,
    {
        let processor = CompletionStreamProcessor::new();
        let adapter = StreamAdapter::new(stream, processor);
        adapter.process_all().await
    }

    /// Process a completion chunk stream with custom processor
    pub async fn process_completion_stream_with_processor<S>(
        stream: S,
        processor: CompletionStreamProcessor,
    ) -> Result<(), std::io::Error>
    where
        S: Stream<Item = CompletionChunk> + Unpin,
    {
        let adapter = StreamAdapter::new(stream, processor);
        adapter.process_all().await
    }

    /// Aggregate completion stream without real-time output
    pub async fn aggregate_completion_stream<S>(stream: S) -> StreamingAggregator
    where
        S: Stream<Item = CompletionChunk> + Unpin,
    {
        let mut aggregator = StreamingAggregator::new();
        
        futures::pin_mut!(stream);
        while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
            aggregator.process_chunk(chunk);
            if aggregator.is_complete() {
                break;
            }
        }
        
        aggregator
    }

    /// Convert AsyncStream to futures Stream for compatibility
    pub fn async_stream_to_stream<T: crate::async_task::NotResult>(
        async_stream: AsyncStream<T>,
    ) -> impl Stream<Item = T> + Unpin {
        struct AsyncStreamAdapter<T: crate::async_task::NotResult> {
            inner: AsyncStream<T>,
        }

        impl<T: crate::async_task::NotResult> Stream for AsyncStreamAdapter<T> {
            type Item = T;

            fn poll_next(
                mut self: std::pin::Pin<&mut Self>,
                cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Self::Item>> {
                use std::pin::Pin;
                Pin::new(&mut self.inner).poll_next(cx)
            }
        }

        AsyncStreamAdapter { inner: async_stream }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn test_streaming_output() {
        let mut output = StreamingOutput::plain();
        output.output_chunk("Hello");
        output.output_chunk(" ");
        output.output_chunk("World!");
        output.newline();
    }

    #[tokio::test]
    async fn test_text_stream_processor() {
        let mut processor = TextStreamProcessor::new();

        assert!(processor.process_item("Hello".to_string()).is_ok());
        assert!(processor.process_item(" World".to_string()).is_ok());
        assert!(processor.finish().is_ok());
    }

    #[tokio::test]
    async fn test_stream_adapter() {
        let text_stream = stream::iter(vec![
            "Hello".to_string(),
            " ".to_string(),
            "World!".to_string(),
        ]);

        let processor = TextStreamProcessor::new();
        let adapter = StreamAdapter::new(text_stream, processor);

        assert!(adapter.process_all().await.is_ok());
    }
}
