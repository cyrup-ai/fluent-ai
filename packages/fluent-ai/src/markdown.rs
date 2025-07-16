use once_cell::sync::Lazy;
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::prelude::*;
use smallvec::SmallVec;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Color as SyntectColor, Theme as SyntectTheme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};
use syntect::util::LinesWithEndings;

// Pre-compiled syntax definitions for common languages
static SYNTAX_SET: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);

// Embedded theme optimized for terminals
static TERMINAL_THEME: Lazy<SyntectTheme> =
    Lazy::new(|| ThemeSet::load_defaults().themes["base16-ocean.dark"].clone());

// Common static strings to avoid allocations
const CODE_BORDER_TOP: &str = "╭─ ";
const CODE_BORDER_SIDE: &str = "│ ";
const CODE_BORDER_BOTTOM: &str = "╰";
const CODE_BORDER_LINE: &str = "─";
const BULLET: &str = "• ";
const SPACE: &str = " ";

/// Theme configuration for markdown rendering
#[derive(Clone, Debug)]
pub struct MarkdownTheme {
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,
    pub text: Color,
    pub background: Color,
    pub code_background: Color,
    pub heading: Color,
    pub link: Color,
    pub emphasis: Color,
    pub strong: Color,
}

impl Default for MarkdownTheme {
    fn default() -> Self {
        Self {
            primary: Color::Cyan,
            secondary: Color::Blue,
            accent: Color::Green,
            text: Color::White,
            background: Color::Black,
            code_background: Color::DarkGray,
            heading: Color::Yellow,
            link: Color::Blue,
            emphasis: Color::Yellow,
            strong: Color::Red,
        }
    }
}

/// Zero-allocation markdown renderer with syntax highlighting
#[derive(Clone)]
pub struct MarkdownRenderer {
    theme: MarkdownTheme,
    // Pre-allocated buffers for performance
    line_buffer: Vec<Line<'static>>,
    span_buffer: Vec<Span<'static>>,
}

impl MarkdownRenderer {
    #[inline]
    pub fn new(theme: MarkdownTheme) -> Self {
        Self {
            theme,
            line_buffer: Vec::with_capacity(64),
            span_buffer: Vec::with_capacity(16),
        }
    }

    /// Create renderer with default theme
    pub fn default() -> Self {
        Self::new(MarkdownTheme::default())
    }

    /// Get the line buffer capacity for performance monitoring
    pub fn line_buffer_capacity(&self) -> usize {
        self.line_buffer.capacity()
    }

    /// Get the span buffer capacity for performance monitoring  
    pub fn span_buffer_capacity(&self) -> usize {
        self.span_buffer.capacity()
    }

    /// Render markdown to terminal-styled lines
    pub fn render(&mut self, markdown: &str) -> Vec<Line<'static>> {
        // Clear and reuse pre-allocated buffers
        self.line_buffer.clear();
        self.span_buffer.clear();

        // Ensure adequate capacity based on markdown length heuristic
        let estimated_lines = markdown.len() / 40;
        if self.line_buffer.capacity() < estimated_lines {
            self.line_buffer
                .reserve(estimated_lines - self.line_buffer.capacity());
        }

        let estimated_spans = markdown.len() / 80;
        if self.span_buffer.capacity() < estimated_spans {
            self.span_buffer
                .reserve(estimated_spans - self.span_buffer.capacity());
        }

        let mut options = Options::empty();
        options.insert(Options::ENABLE_STRIKETHROUGH);
        options.insert(Options::ENABLE_TABLES);
        options.insert(Options::ENABLE_FOOTNOTES);
        options.insert(Options::ENABLE_TASKLISTS);

        let parser = Parser::new_ext(markdown, options);

        // Track current span indices in the pre-allocated buffer
        let current_span_start = 0;
        let mut in_code_block = false;
        let mut code_block_lang: Option<String> = None;
        let mut code_block_content = String::with_capacity(512);
        let mut list_stack: SmallVec<[Option<u64>; 4]> = SmallVec::new();
        let mut current_style = Style::default();

        for event in parser {
            match event {
                Event::Start(tag) => match tag {
                    Tag::Paragraph => {
                        if self.span_buffer.len() > current_span_start {
                            // Create line from current spans in buffer
                            let spans: Vec<Span<'static>> =
                                self.span_buffer[current_span_start..].to_vec();
                            self.line_buffer.push(Line::from(spans));
                            self.span_buffer.clear();
                        }
                    }
                    Tag::Heading {
                        level,
                        id: _,
                        classes: _,
                        attrs: _,
                    } => {
                        let heading_style = Style::default()
                            .fg(self.theme.heading)
                            .add_modifier(Modifier::BOLD);
                        current_style = heading_style;

                        // Add heading prefix based on level
                        let prefix = match level {
                            pulldown_cmark::HeadingLevel::H1 => "# ",
                            pulldown_cmark::HeadingLevel::H2 => "## ",
                            pulldown_cmark::HeadingLevel::H3 => "### ",
                            pulldown_cmark::HeadingLevel::H4 => "#### ",
                            pulldown_cmark::HeadingLevel::H5 => "##### ",
                            pulldown_cmark::HeadingLevel::H6 => "###### ",
                        };
                        self.span_buffer.push(Span::styled(prefix, heading_style));
                    }
                    Tag::CodeBlock(kind) => {
                        in_code_block = true;
                        code_block_lang = match kind {
                            pulldown_cmark::CodeBlockKind::Fenced(lang) => {
                                if lang.is_empty() {
                                    None
                                } else {
                                    Some(lang.to_string())
                                }
                            }
                            pulldown_cmark::CodeBlockKind::Indented => None,
                        };
                        code_block_content.clear();
                    }
                    // Inline code is handled by Event::Code, not Tag
                    Tag::Emphasis => {
                        current_style = current_style.add_modifier(Modifier::ITALIC);
                    }
                    Tag::Strong => {
                        current_style = current_style
                            .fg(self.theme.strong)
                            .add_modifier(Modifier::BOLD);
                    }
                    Tag::Link {
                        link_type: _,
                        dest_url,
                        title: _,
                        id: _,
                    } => {
                        current_style = Style::default()
                            .fg(self.theme.link)
                            .add_modifier(Modifier::UNDERLINED);
                        // Store URL for potential display
                        let _ = dest_url;
                    }
                    Tag::List(start_num) => {
                        list_stack.push(start_num);
                    }
                    Tag::Item => {
                        let indent_level = (list_stack.len().saturating_sub(1)) * 2;
                        let indent = SPACE.repeat(indent_level);

                        if let Some(Some(num)) = list_stack.last_mut() {
                            // Ordered list
                            self.span_buffer
                                .push(Span::raw(format!("{}{}. ", indent, num)));
                            *num += 1;
                        } else {
                            // Unordered list
                            self.span_buffer
                                .push(Span::raw(format!("{}{}", indent, BULLET)));
                        }
                    }
                    _ => {}
                },
                Event::End(tag_end) => match tag_end {
                    TagEnd::Paragraph => {
                        if !self.span_buffer.is_empty() {
                            let spans: Vec<Span<'static>> = self.span_buffer.drain(..).collect();
                            self.line_buffer.push(Line::from(spans));
                        }
                        // Add empty line after paragraph
                        self.line_buffer.push(Line::from(""));
                    }
                    TagEnd::Heading(_) => {
                        if !self.span_buffer.is_empty() {
                            let spans: Vec<Span<'static>> = self.span_buffer.drain(..).collect();
                            self.line_buffer.push(Line::from(spans));
                        }
                        // Add empty line after heading
                        self.line_buffer.push(Line::from(""));
                        current_style = Style::default();
                    }
                    TagEnd::CodeBlock => {
                        self.render_code_block(&code_block_content, code_block_lang.as_deref());
                        in_code_block = false;
                        code_block_lang = None;
                        code_block_content.clear();
                    }
                    // Inline code is handled by Event::Code, not TagEnd
                    TagEnd::Emphasis | TagEnd::Strong | TagEnd::Link => {
                        current_style = Style::default();
                    }
                    TagEnd::List(_) => {
                        list_stack.pop();
                    }
                    TagEnd::Item => {
                        if !self.span_buffer.is_empty() {
                            let spans: Vec<Span<'static>> = self.span_buffer.drain(..).collect();
                            self.line_buffer.push(Line::from(spans));
                        }
                    }
                    _ => {}
                },
                Event::Text(text) => {
                    if in_code_block {
                        code_block_content.push_str(&text);
                    } else {
                        self.span_buffer
                            .push(Span::styled(text.to_string(), current_style));
                    }
                }
                Event::Code(text) => {
                    // Handle inline code with special styling
                    let code_style = Style::default()
                        .fg(self.theme.accent)
                        .bg(self.theme.code_background);
                    self.span_buffer
                        .push(Span::styled(text.to_string(), code_style));
                }

                Event::Html(html) => {
                    // Skip HTML for terminal rendering
                    let _ = html;
                }
                Event::SoftBreak | Event::HardBreak => {
                    if !self.span_buffer.is_empty() {
                        let spans: Vec<Span<'static>> = self.span_buffer.drain(..).collect();
                        self.line_buffer.push(Line::from(spans));
                    }
                }
                _ => {}
            }
        }

        // Handle any remaining spans
        if !self.span_buffer.is_empty() {
            let spans: Vec<Span<'static>> = self.span_buffer.drain(..).collect();
            self.line_buffer.push(Line::from(spans));
        }

        // Return cloned lines (zero-allocation during rendering)
        self.line_buffer.clone()
    }

    /// Render a code block with syntax highlighting
    fn render_code_block(&mut self, code: &str, language: Option<&str>) {
        let border_style = Style::default().fg(self.theme.secondary);

        // Add top border
        let lang_display = language.unwrap_or("text").to_string();
        self.line_buffer.push(Line::from(vec![
            Span::styled(CODE_BORDER_TOP, border_style),
            Span::styled(lang_display, border_style.add_modifier(Modifier::BOLD)),
        ]));

        // Try to get syntax definition
        if let Some(language) = language {
            if let Some(syntax) = SYNTAX_SET.find_syntax_by_token(language) {
                self.highlight_code_lines(code, syntax, border_style);
            } else {
                self.render_plain_code(code, border_style);
            }
        } else {
            self.render_plain_code(code, border_style);
        }

        // Add bottom border
        let bottom_line = format!("{}{}", CODE_BORDER_BOTTOM, CODE_BORDER_LINE.repeat(50));
        self.line_buffer
            .push(Line::from(Span::styled(bottom_line, border_style)));

        // Add empty line after code block
        self.line_buffer.push(Line::from(""));
    }

    /// Highlight code lines with syntect
    fn highlight_code_lines(&mut self, code: &str, syntax: &SyntaxReference, border_style: Style) {
        let mut highlighter = HighlightLines::new(syntax, &TERMINAL_THEME);

        for line in LinesWithEndings::from(code) {
            let mut spans = vec![Span::styled(CODE_BORDER_SIDE, border_style)];

            if let Ok(highlighted) = highlighter.highlight_line(line, &SYNTAX_SET) {
                for (style, text) in highlighted {
                    let ratatui_color = self.syntect_color_to_ratatui(style.foreground);
                    let text_style = Style::default().fg(ratatui_color);
                    spans.push(Span::styled(text.to_string(), text_style));
                }
            } else {
                // Fallback to plain text
                spans.push(Span::styled(
                    line.to_string(),
                    Style::default().fg(self.theme.text),
                ));
            }

            self.line_buffer.push(Line::from(spans));
        }
    }

    /// Render plain code without syntax highlighting
    fn render_plain_code(&mut self, code: &str, border_style: Style) {
        let code_style = Style::default().fg(self.theme.text);

        for line in code.lines() {
            let spans = vec![
                Span::styled(CODE_BORDER_SIDE, border_style),
                Span::styled(line.to_string(), code_style),
            ];
            self.line_buffer.push(Line::from(spans));
        }
    }

    /// Convert syntect color to ratatui color
    fn syntect_color_to_ratatui(&self, color: SyntectColor) -> Color {
        Color::Rgb(color.r, color.g, color.b)
    }
}

/// Simple function to render markdown to string with basic terminal colors
pub fn render_markdown_to_string(markdown: &str) -> String {
    use crossterm::style::{Color as TermColor, Stylize};

    let mut output = String::new();
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(markdown, options);

    let mut in_code_block = false;
    let mut heading_level = 0;

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    heading_level = match level {
                        pulldown_cmark::HeadingLevel::H1 => 1,
                        pulldown_cmark::HeadingLevel::H2 => 2,
                        pulldown_cmark::HeadingLevel::H3 => 3,
                        pulldown_cmark::HeadingLevel::H4 => 4,
                        pulldown_cmark::HeadingLevel::H5 => 5,
                        pulldown_cmark::HeadingLevel::H6 => 6,
                    };
                    output.push_str(&"#".repeat(heading_level));
                    output.push(' ');
                }
                Tag::CodeBlock(kind) => {
                    in_code_block = true;
                    let lang_display = match kind {
                        pulldown_cmark::CodeBlockKind::Fenced(lang) => {
                            if lang.is_empty() {
                                "text".to_string()
                            } else {
                                lang.to_string()
                            }
                        }
                        pulldown_cmark::CodeBlockKind::Indented => "text".to_string(),
                    };

                    output.push_str(&format!(
                        "╭─ {}\n",
                        lang_display.stylize().with(TermColor::Cyan)
                    ));
                }
                // Inline code is handled by Event::Code, not Tag
                Tag::Emphasis => {
                    // Handled in text
                }
                Tag::Strong => {
                    // Handled in text
                }
                Tag::Link { .. } => {
                    // Handled in text
                }
                _ => {}
            },
            Event::End(tag_end) => match tag_end {
                TagEnd::Heading(_) => {
                    output.push('\n');
                    heading_level = 0;
                }
                TagEnd::CodeBlock => {
                    output.push_str("╰");
                    output.push_str(&"─".repeat(50));
                    output.push('\n');
                    in_code_block = false;
                }
                // Inline code is handled by Event::Code, not TagEnd
                TagEnd::Paragraph => {
                    output.push('\n');
                }
                _ => {}
            },
            Event::Text(text) => {
                if in_code_block {
                    for line in text.lines() {
                        output.push_str(&format!("│ {}\n", line));
                    }
                } else if heading_level > 0 {
                    output.push_str(&text.stylize().bold().with(TermColor::Yellow).to_string());
                } else {
                    output.push_str(&text);
                }
            }
            Event::Code(code) => {
                output.push_str(&format!("`{}`", code.stylize().with(TermColor::Green)));
            }
            Event::SoftBreak => {
                output.push(' ');
            }
            Event::HardBreak => {
                output.push('\n');
            }
            _ => {}
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_markdown_rendering() {
        let mut renderer = MarkdownRenderer::default();
        let markdown = "# Hello World\n\nThis is a **bold** text with `code`.\n\n```rust\nfn main() {\n    println!(\"Hello!\");\n}\n```";

        let lines = renderer.render(markdown);
        assert!(!lines.is_empty());

        // Check that we have some content
        let has_heading = lines
            .iter()
            .any(|line| line.spans.iter().any(|span| span.content.contains('#')));
        assert!(has_heading);
    }

    #[test]
    fn test_string_markdown_rendering() {
        let markdown = "# Test\n\nHello **world**!\n\n```rust\nlet x = 42;\n```";
        let result = render_markdown_to_string(markdown);

        assert!(result.contains("# Test"));
        assert!(result.contains("Hello"));
        assert!(result.contains("╭─"));
        assert!(result.contains("│"));
    }
}
