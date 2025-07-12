use futures::stream::Stream;

use crate::markdown::{render_markdown_to_string, MarkdownRenderer, MarkdownTheme};

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
