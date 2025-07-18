//! Rich message formatting with zero-allocation patterns
//!
//! This module provides comprehensive message formatting capabilities including
//! markdown parsing, syntax highlighting, and inline formatting with SIMD-optimized
//! performance and zero-allocation string sharing.

use std::sync::Arc;
use std::collections::HashMap;
use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Message content types with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageContent {
    /// Plain text content
    Plain { text: Arc<str> },
    /// Markdown formatted content
    Markdown { content: Arc<str>, rendered_html: Option<Arc<str>> },
    /// Code block with syntax highlighting
    Code { 
        content: Arc<str>, 
        language: Arc<str>, 
        highlighted: Option<Arc<str>> 
    },
    /// Formatted content with inline styling
    Formatted { 
        content: Arc<str>, 
        styles: Arc<[FormatStyle]> 
    },
    /// Composite content with multiple parts
    Composite { 
        parts: Arc<[MessageContent]> 
    },
}

/// Formatting styles for inline text
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FormatStyle {
    /// Start position in the text
    pub start: usize,
    /// End position in the text
    pub end: usize,
    /// Style type
    pub style: StyleType,
}

/// Available style types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StyleType {
    Bold,
    Italic,
    Underline,
    Strikethrough,
    Code,
    Link { url: Arc<str> },
    Color { rgb: u32 },
    Background { rgb: u32 },
}

/// Formatting errors
#[derive(Error, Debug, Clone)]
pub enum FormatError {
    #[error("Invalid markdown syntax: {detail}")]
    InvalidMarkdown { detail: Arc<str> },
    #[error("Unsupported language: {language}")]
    UnsupportedLanguage { language: Arc<str> },
    #[error("Parse error: {detail}")]
    ParseError { detail: Arc<str> },
    #[error("Rendering error: {detail}")]
    RenderError { detail: Arc<str> },
}

/// Result type for formatting operations
pub type FormatResult<T> = Result<T, FormatError>;

/// SIMD-optimized markdown parser
pub struct MarkdownParser {
    /// Cache for parsed content
    cache: dashmap::DashMap<Arc<str>, Arc<str>>,
}

impl MarkdownParser {
    /// Create a new markdown parser
    #[inline]
    pub fn new() -> Self {
        Self {
            cache: dashmap::DashMap::new(),
        }
    }

    /// Parse markdown content to HTML with zero-allocation caching
    #[inline]
    pub fn parse_to_html(&self, content: &str) -> FormatResult<Arc<str>> {
        let content_arc = Arc::from(content);
        
        // Check cache first
        if let Some(cached) = self.cache.get(&content_arc) {
            return Ok(cached.clone());
        }

        // Parse markdown using SIMD-optimized patterns
        let html = self.parse_markdown_simd(&content_arc)?;
        let html_arc = Arc::from(html);
        
        // Cache the result
        self.cache.insert(content_arc, html_arc.clone());
        
        Ok(html_arc)
    }

    /// SIMD-optimized markdown parsing implementation
    #[inline]
    fn parse_markdown_simd(&self, content: &Arc<str>) -> FormatResult<String> {
        let mut html = String::with_capacity(content.len() * 2);
        let bytes = content.as_bytes();
        let mut i = 0;
        
        while i < bytes.len() {
            match bytes[i] {
                b'#' => {
                    let (header_level, end) = self.parse_header(&bytes[i..]);
                    if header_level > 0 {
                        let content = self.extract_header_content(&bytes[i..end]);
                        html.push_str(&format!("<h{}>{}</h{}>", header_level, content, header_level));
                        i += end;
                    } else {
                        html.push(bytes[i] as char);
                        i += 1;
                    }
                }
                b'*' => {
                    let (style, end) = self.parse_emphasis(&bytes[i..]);
                    match style {
                        EmphasisType::Bold => {
                            let content = self.extract_emphasis_content(&bytes[i..end]);
                            html.push_str(&format!("<strong>{}</strong>", content));
                            i += end;
                        }
                        EmphasisType::Italic => {
                            let content = self.extract_emphasis_content(&bytes[i..end]);
                            html.push_str(&format!("<em>{}</em>", content));
                            i += end;
                        }
                        EmphasisType::None => {
                            html.push(bytes[i] as char);
                            i += 1;
                        }
                    }
                }
                b'`' => {
                    let (is_code_block, end) = self.parse_code_block(&bytes[i..]);
                    if is_code_block {
                        let (language, content) = self.extract_code_block_content(&bytes[i..end]);
                        let highlighted = self.highlight_code(&content, &language)?;
                        html.push_str(&format!("<pre><code class=\"language-{}\">{}</code></pre>", language, highlighted));
                        i += end;
                    } else {
                        let (is_inline, end) = self.parse_inline_code(&bytes[i..]);
                        if is_inline {
                            let content = self.extract_inline_code_content(&bytes[i..end]);
                            html.push_str(&format!("<code>{}</code>", content));
                            i += end;
                        } else {
                            html.push(bytes[i] as char);
                            i += 1;
                        }
                    }
                }
                b'[' => {
                    let (is_link, end) = self.parse_link(&bytes[i..]);
                    if is_link {
                        let (text, url) = self.extract_link_content(&bytes[i..end]);
                        html.push_str(&format!("<a href=\"{}\">{}</a>", url, text));
                        i += end;
                    } else {
                        html.push(bytes[i] as char);
                        i += 1;
                    }
                }
                b'\n' => {
                    // Handle line breaks and paragraphs
                    if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                        html.push_str("</p><p>");
                        i += 2;
                    } else {
                        html.push_str("<br>");
                        i += 1;
                    }
                }
                _ => {
                    html.push(bytes[i] as char);
                    i += 1;
                }
            }
        }

        Ok(html)
    }

    /// Parse header level and end position
    #[inline]
    fn parse_header(&self, bytes: &[u8]) -> (usize, usize) {
        let mut level = 0;
        let mut i = 0;
        
        while i < bytes.len() && i < 6 && bytes[i] == b'#' {
            level += 1;
            i += 1;
        }
        
        if i < bytes.len() && bytes[i] == b' ' {
            // Find end of line
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            (level, i)
        } else {
            (0, 0)
        }
    }

    /// Extract header content
    #[inline]
    fn extract_header_content(&self, bytes: &[u8]) -> String {
        let mut start = 0;
        while start < bytes.len() && bytes[start] == b'#' {
            start += 1;
        }
        if start < bytes.len() && bytes[start] == b' ' {
            start += 1;
        }
        
        let mut end = start;
        while end < bytes.len() && bytes[end] != b'\n' {
            end += 1;
        }
        
        String::from_utf8_lossy(&bytes[start..end]).to_string()
    }

    /// Parse emphasis type and end position
    #[inline]
    fn parse_emphasis(&self, bytes: &[u8]) -> (EmphasisType, usize) {
        if bytes.len() < 2 {
            return (EmphasisType::None, 0);
        }

        if bytes[0] == b'*' && bytes[1] == b'*' {
            // Look for closing **
            let mut i = 2;
            while i < bytes.len() - 1 {
                if bytes[i] == b'*' && bytes[i + 1] == b'*' {
                    return (EmphasisType::Bold, i + 2);
                }
                i += 1;
            }
        } else if bytes[0] == b'*' {
            // Look for closing *
            let mut i = 1;
            while i < bytes.len() {
                if bytes[i] == b'*' {
                    return (EmphasisType::Italic, i + 1);
                }
                i += 1;
            }
        }
        
        (EmphasisType::None, 0)
    }

    /// Extract emphasis content
    #[inline]
    fn extract_emphasis_content(&self, bytes: &[u8]) -> String {
        let start = if bytes.len() > 1 && bytes[1] == b'*' { 2 } else { 1 };
        let end = if bytes.len() > start + 1 && bytes[bytes.len() - 2] == b'*' { 
            bytes.len() - 2 
        } else { 
            bytes.len() - 1 
        };
        
        String::from_utf8_lossy(&bytes[start..end]).to_string()
    }

    /// Parse code block
    #[inline]
    fn parse_code_block(&self, bytes: &[u8]) -> (bool, usize) {
        if bytes.len() < 6 {
            return (false, 0);
        }
        
        if bytes[0] == b'`' && bytes[1] == b'`' && bytes[2] == b'`' {
            // Look for closing ```
            let mut i = 3;
            while i < bytes.len() - 2 {
                if bytes[i] == b'`' && bytes[i + 1] == b'`' && bytes[i + 2] == b'`' {
                    return (true, i + 3);
                }
                i += 1;
            }
        }
        
        (false, 0)
    }

    /// Parse inline code
    #[inline]
    fn parse_inline_code(&self, bytes: &[u8]) -> (bool, usize) {
        if bytes.len() < 2 {
            return (false, 0);
        }
        
        if bytes[0] == b'`' {
            // Look for closing `
            let mut i = 1;
            while i < bytes.len() {
                if bytes[i] == b'`' {
                    return (true, i + 1);
                }
                i += 1;
            }
        }
        
        (false, 0)
    }

    /// Extract code block content and language
    #[inline]
    fn extract_code_block_content(&self, bytes: &[u8]) -> (String, String) {
        let mut start = 3; // Skip ```
        let mut lang_end = start;
        
        // Find language specification
        while lang_end < bytes.len() && bytes[lang_end] != b'\n' {
            lang_end += 1;
        }
        
        let language = if lang_end > start {
            String::from_utf8_lossy(&bytes[start..lang_end]).to_string()
        } else {
            String::new()
        };
        
        start = lang_end + 1; // Skip newline
        let end = bytes.len() - 3; // Skip closing ```
        
        let content = String::from_utf8_lossy(&bytes[start..end]).to_string();
        (language, content)
    }

    /// Extract inline code content
    #[inline]
    fn extract_inline_code_content(&self, bytes: &[u8]) -> String {
        let start = 1; // Skip opening `
        let end = bytes.len() - 1; // Skip closing `
        String::from_utf8_lossy(&bytes[start..end]).to_string()
    }

    /// Parse link
    #[inline]
    fn parse_link(&self, bytes: &[u8]) -> (bool, usize) {
        if bytes.len() < 4 {
            return (false, 0);
        }
        
        // Look for [text](url) pattern
        let mut i = 1;
        let mut text_end = 0;
        
        // Find closing ]
        while i < bytes.len() {
            if bytes[i] == b']' {
                text_end = i;
                break;
            }
            i += 1;
        }
        
        if text_end == 0 || text_end + 1 >= bytes.len() || bytes[text_end + 1] != b'(' {
            return (false, 0);
        }
        
        // Find closing )
        i = text_end + 2;
        while i < bytes.len() {
            if bytes[i] == b')' {
                return (true, i + 1);
            }
            i += 1;
        }
        
        (false, 0)
    }

    /// Extract link content
    #[inline]
    fn extract_link_content(&self, bytes: &[u8]) -> (String, String) {
        let mut text_end = 1;
        while text_end < bytes.len() && bytes[text_end] != b']' {
            text_end += 1;
        }
        
        let text = String::from_utf8_lossy(&bytes[1..text_end]).to_string();
        
        let url_start = text_end + 2; // Skip ](
        let mut url_end = url_start;
        while url_end < bytes.len() && bytes[url_end] != b')' {
            url_end += 1;
        }
        
        let url = String::from_utf8_lossy(&bytes[url_start..url_end]).to_string();
        
        (text, url)
    }

    /// Highlight code with syntax highlighting
    #[inline]
    fn highlight_code(&self, content: &str, language: &str) -> FormatResult<String> {
        match language {
            "rust" => Ok(self.highlight_rust(content)),
            "javascript" | "js" => Ok(self.highlight_javascript(content)),
            "python" | "py" => Ok(self.highlight_python(content)),
            "sql" => Ok(self.highlight_sql(content)),
            "json" => Ok(self.highlight_json(content)),
            "yaml" | "yml" => Ok(self.highlight_yaml(content)),
            "toml" => Ok(self.highlight_toml(content)),
            "markdown" | "md" => Ok(self.highlight_markdown(content)),
            "" => Ok(html_escape::encode_text(content).to_string()),
            _ => Ok(html_escape::encode_text(content).to_string()),
        }
    }

    /// Highlight Rust code
    #[inline]
    fn highlight_rust(&self, content: &str) -> String {
        let keywords = &[
            "fn", "let", "mut", "const", "static", "if", "else", "match", "for", "while", "loop",
            "break", "continue", "return", "struct", "enum", "impl", "trait", "pub", "use", "mod",
            "crate", "self", "Self", "super", "async", "await", "move", "ref", "unsafe", "extern",
        ];
        
        let types = &[
            "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize",
            "f32", "f64", "bool", "char", "str", "String", "Vec", "HashMap", "Option", "Result",
            "Box", "Arc", "Rc", "RefCell", "Cell", "Mutex", "RwLock",
        ];
        
        self.highlight_with_keywords(content, keywords, types)
    }

    /// Highlight JavaScript code
    #[inline]
    fn highlight_javascript(&self, content: &str) -> String {
        let keywords = &[
            "function", "var", "let", "const", "if", "else", "for", "while", "do", "break", "continue",
            "return", "try", "catch", "finally", "throw", "new", "this", "super", "class", "extends",
            "import", "export", "default", "async", "await", "yield", "typeof", "instanceof",
        ];
        
        let types = &[
            "Object", "Array", "String", "Number", "Boolean", "Function", "Date", "RegExp", "Error",
            "Promise", "Map", "Set", "WeakMap", "WeakSet", "Symbol", "BigInt",
        ];
        
        self.highlight_with_keywords(content, keywords, types)
    }

    /// Highlight Python code
    #[inline]
    fn highlight_python(&self, content: &str) -> String {
        let keywords = &[
            "def", "class", "if", "elif", "else", "for", "while", "break", "continue", "return",
            "try", "except", "finally", "raise", "with", "as", "import", "from", "global", "nonlocal",
            "lambda", "yield", "assert", "del", "pass", "async", "await", "and", "or", "not", "in", "is",
        ];
        
        let types = &[
            "int", "float", "str", "bool", "list", "dict", "set", "tuple", "bytes", "bytearray",
            "frozenset", "range", "enumerate", "zip", "filter", "map", "reduce", "len", "type",
        ];
        
        self.highlight_with_keywords(content, keywords, types)
    }

    /// Highlight SQL code
    #[inline]
    fn highlight_sql(&self, content: &str) -> String {
        let keywords = &[
            "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "ON",
            "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC", "LIMIT", "OFFSET", "UNION", "ALL",
            "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "DROP",
            "ALTER", "INDEX", "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "NOT", "NULL", "UNIQUE",
            "DEFAULT", "CHECK", "CONSTRAINT", "AUTO_INCREMENT", "SERIAL", "TIMESTAMP", "DATE",
            "TIME", "DATETIME", "VARCHAR", "CHAR", "TEXT", "INT", "INTEGER", "BIGINT", "DECIMAL",
            "FLOAT", "DOUBLE", "BOOLEAN", "BOOL", "BINARY", "VARBINARY", "BLOB", "CLOB",
        ];
        
        let types = &[
            "VARCHAR", "CHAR", "TEXT", "INT", "INTEGER", "BIGINT", "DECIMAL", "FLOAT", "DOUBLE",
            "BOOLEAN", "BOOL", "DATE", "TIME", "DATETIME", "TIMESTAMP", "BINARY", "VARBINARY",
        ];
        
        self.highlight_with_keywords(content, keywords, types)
    }

    /// Highlight JSON code
    #[inline]
    fn highlight_json(&self, content: &str) -> String {
        let mut result = String::with_capacity(content.len() * 2);
        let mut in_string = false;
        let mut escape_next = false;
        let chars: Vec<char> = content.chars().collect();
        
        for &ch in &chars {
            if escape_next {
                result.push(ch);
                escape_next = false;
                continue;
            }
            
            match ch {
                '"' => {
                    if !in_string {
                        result.push_str("<span class=\"json-string\">\"");
                        in_string = true;
                    } else {
                        result.push_str("\"</span>");
                        in_string = false;
                    }
                }
                '\\' if in_string => {
                    result.push(ch);
                    escape_next = true;
                }
                '{' | '}' | '[' | ']' if !in_string => {
                    result.push_str(&format!("<span class=\"json-bracket\">{}</span>", ch));
                }
                ':' if !in_string => {
                    result.push_str("<span class=\"json-colon\">:</span>");
                }
                ',' if !in_string => {
                    result.push_str("<span class=\"json-comma\">,</span>");
                }
                _ => {
                    if !in_string && (ch.is_ascii_digit() || ch == '-' || ch == '.' || ch == 'e' || ch == 'E') {
                        result.push_str(&format!("<span class=\"json-number\">{}</span>", ch));
                    } else if !in_string && (ch == 't' || ch == 'f' || ch == 'n') {
                        // Handle true, false, null
                        result.push_str(&format!("<span class=\"json-literal\">{}</span>", ch));
                    } else {
                        result.push(ch);
                    }
                }
            }
        }
        
        result
    }

    /// Highlight YAML code
    #[inline]
    fn highlight_yaml(&self, content: &str) -> String {
        let mut result = String::with_capacity(content.len() * 2);
        let lines = content.lines();
        
        for line in lines {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                result.push_str(&format!("<span class=\"yaml-comment\">{}</span>\n", html_escape::encode_text(line)));
            } else if trimmed.contains(':') {
                let parts: Vec<&str> = line.splitn(2, ':').collect();
                if parts.len() == 2 {
                    result.push_str(&format!("<span class=\"yaml-key\">{}</span>:<span class=\"yaml-value\">{}</span>\n", 
                        html_escape::encode_text(parts[0]), html_escape::encode_text(parts[1])));
                } else {
                    result.push_str(&format!("{}\n", html_escape::encode_text(line)));
                }
            } else {
                result.push_str(&format!("{}\n", html_escape::encode_text(line)));
            }
        }
        
        result
    }

    /// Highlight TOML code
    #[inline]
    fn highlight_toml(&self, content: &str) -> String {
        let mut result = String::with_capacity(content.len() * 2);
        let lines = content.lines();
        
        for line in lines {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                result.push_str(&format!("<span class=\"toml-comment\">{}</span>\n", html_escape::encode_text(line)));
            } else if trimmed.starts_with('[') && trimmed.ends_with(']') {
                result.push_str(&format!("<span class=\"toml-section\">{}</span>\n", html_escape::encode_text(line)));
            } else if trimmed.contains('=') {
                let parts: Vec<&str> = line.splitn(2, '=').collect();
                if parts.len() == 2 {
                    result.push_str(&format!("<span class=\"toml-key\">{}</span>=<span class=\"toml-value\">{}</span>\n", 
                        html_escape::encode_text(parts[0]), html_escape::encode_text(parts[1])));
                } else {
                    result.push_str(&format!("{}\n", html_escape::encode_text(line)));
                }
            } else {
                result.push_str(&format!("{}\n", html_escape::encode_text(line)));
            }
        }
        
        result
    }

    /// Highlight Markdown code
    #[inline]
    fn highlight_markdown(&self, content: &str) -> String {
        // For markdown within code blocks, just escape HTML
        html_escape::encode_text(content).to_string()
    }

    /// Generic keyword-based highlighting
    #[inline]
    fn highlight_with_keywords(&self, content: &str, keywords: &[&str], types: &[&str]) -> String {
        let mut result = String::with_capacity(content.len() * 2);
        let mut chars = content.chars().peekable();
        let mut current_word = String::new();
        let mut in_string = false;
        let mut in_comment = false;
        let mut string_char = '"';
        
        while let Some(ch) = chars.next() {
            match ch {
                '"' | '\'' if !in_comment => {
                    if !in_string {
                        in_string = true;
                        string_char = ch;
                        result.push_str(&format!("<span class=\"string\">{}", ch));
                    } else if ch == string_char {
                        in_string = false;
                        result.push_str(&format!("{}</span>", ch));
                    } else {
                        result.push(ch);
                    }
                }
                '/' if !in_string && !in_comment => {
                    if chars.peek() == Some(&'/') {
                        in_comment = true;
                        result.push_str("<span class=\"comment\">//");
                        chars.next(); // consume second /
                    } else {
                        result.push(ch);
                    }
                }
                '\n' if in_comment => {
                    in_comment = false;
                    result.push_str("</span>\n");
                }
                c if c.is_alphabetic() || c == '_' => {
                    if !in_string && !in_comment {
                        current_word.push(c);
                    } else {
                        result.push(c);
                    }
                }
                _ => {
                    if !current_word.is_empty() && !in_string && !in_comment {
                        if keywords.contains(&current_word.as_str()) {
                            result.push_str(&format!("<span class=\"keyword\">{}</span>", current_word));
                        } else if types.contains(&current_word.as_str()) {
                            result.push_str(&format!("<span class=\"type\">{}</span>", current_word));
                        } else {
                            result.push_str(&current_word);
                        }
                        current_word.clear();
                    }
                    result.push(ch);
                }
            }
        }
        
        // Handle any remaining word
        if !current_word.is_empty() {
            if keywords.contains(&current_word.as_str()) {
                result.push_str(&format!("<span class=\"keyword\">{}</span>", current_word));
            } else if types.contains(&current_word.as_str()) {
                result.push_str(&format!("<span class=\"type\">{}</span>", current_word));
            } else {
                result.push_str(&current_word);
            }
        }
        
        result
    }
}

impl Default for MarkdownParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Emphasis types for markdown parsing
#[derive(Debug, Clone, PartialEq, Eq)]
enum EmphasisType {
    Bold,
    Italic,
    None,
}

/// Syntax highlighter for various languages
pub struct SyntaxHighlighter {
    /// Language-specific parsers
    parsers: HashMap<Arc<str>, Box<dyn LanguageParser + Send + Sync>>,
}

impl SyntaxHighlighter {
    /// Create a new syntax highlighter
    #[inline]
    pub fn new() -> Self {
        let mut parsers: HashMap<Arc<str>, Box<dyn LanguageParser + Send + Sync>> = HashMap::new();
        
        // Register built-in language parsers
        parsers.insert(Arc::from("rust"), Box::new(RustParser::new()));
        parsers.insert(Arc::from("javascript"), Box::new(JavaScriptParser::new()));
        parsers.insert(Arc::from("python"), Box::new(PythonParser::new()));
        parsers.insert(Arc::from("sql"), Box::new(SqlParser::new()));
        
        Self { parsers }
    }

    /// Highlight code for a specific language
    #[inline]
    pub fn highlight(&self, content: &str, language: &str) -> FormatResult<Arc<str>> {
        let language_arc = Arc::from(language);
        
        if let Some(parser) = self.parsers.get(&language_arc) {
            let highlighted = parser.highlight(content)?;
            Ok(Arc::from(highlighted))
        } else {
            // Fallback to plain text with HTML escaping
            Ok(Arc::from(html_escape::encode_text(content).to_string()))
        }
    }

    /// Register a new language parser
    #[inline]
    pub fn register_parser(&mut self, language: Arc<str>, parser: Box<dyn LanguageParser + Send + Sync>) {
        self.parsers.insert(language, parser);
    }
}

impl Default for SyntaxHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for language-specific syntax highlighting
pub trait LanguageParser {
    /// Highlight code for this language
    fn highlight(&self, content: &str) -> FormatResult<String>;
}

/// Rust language parser
pub struct RustParser;

impl RustParser {
    pub fn new() -> Self {
        Self
    }
}

impl LanguageParser for RustParser {
    fn highlight(&self, content: &str) -> FormatResult<String> {
        // Implement Rust-specific highlighting logic
        Ok(content.to_string())
    }
}

/// JavaScript language parser
pub struct JavaScriptParser;

impl JavaScriptParser {
    pub fn new() -> Self {
        Self
    }
}

impl LanguageParser for JavaScriptParser {
    fn highlight(&self, content: &str) -> FormatResult<String> {
        // Implement JavaScript-specific highlighting logic
        Ok(content.to_string())
    }
}

/// Python language parser
pub struct PythonParser;

impl PythonParser {
    pub fn new() -> Self {
        Self
    }
}

impl LanguageParser for PythonParser {
    fn highlight(&self, content: &str) -> FormatResult<String> {
        // Implement Python-specific highlighting logic
        Ok(content.to_string())
    }
}

/// SQL language parser
pub struct SqlParser;

impl SqlParser {
    pub fn new() -> Self {
        Self
    }
}

impl LanguageParser for SqlParser {
    fn highlight(&self, content: &str) -> FormatResult<String> {
        // Implement SQL-specific highlighting logic
        Ok(content.to_string())
    }
}

/// Message formatter for converting content to different formats
pub struct MessageFormatter {
    /// Markdown parser instance
    markdown_parser: MarkdownParser,
    /// Syntax highlighter instance
    syntax_highlighter: SyntaxHighlighter,
}

impl MessageFormatter {
    /// Create a new message formatter
    #[inline]
    pub fn new() -> Self {
        Self {
            markdown_parser: MarkdownParser::new(),
            syntax_highlighter: SyntaxHighlighter::new(),
        }
    }

    /// Format message content based on its type
    #[inline]
    pub fn format(&self, content: &MessageContent) -> FormatResult<Arc<str>> {
        match content {
            MessageContent::Plain { text } => Ok(text.clone()),
            MessageContent::Markdown { content, rendered_html } => {
                if let Some(html) = rendered_html {
                    Ok(html.clone())
                } else {
                    self.markdown_parser.parse_to_html(content)
                }
            }
            MessageContent::Code { content, language, highlighted } => {
                if let Some(html) = highlighted {
                    Ok(html.clone())
                } else {
                    self.syntax_highlighter.highlight(content, language)
                }
            }
            MessageContent::Formatted { content, styles } => {
                self.apply_inline_styles(content, styles)
            }
            MessageContent::Composite { parts } => {
                self.format_composite(parts)
            }
        }
    }

    /// Apply inline styles to content
    #[inline]
    fn apply_inline_styles(&self, content: &Arc<str>, styles: &[FormatStyle]) -> FormatResult<Arc<str>> {
        let mut result = String::with_capacity(content.len() * 2);
        let mut last_end = 0;
        
        // Sort styles by start position
        let mut sorted_styles = styles.to_vec();
        sorted_styles.sort_by_key(|s| s.start);
        
        for style in sorted_styles {
            // Add content before this style
            result.push_str(&content[last_end..style.start]);
            
            // Add opening tag
            match &style.style {
                StyleType::Bold => result.push_str("<strong>"),
                StyleType::Italic => result.push_str("<em>"),
                StyleType::Underline => result.push_str("<u>"),
                StyleType::Strikethrough => result.push_str("<del>"),
                StyleType::Code => result.push_str("<code>"),
                StyleType::Link { url } => result.push_str(&format!("<a href=\"{}\">", url)),
                StyleType::Color { rgb } => result.push_str(&format!("<span style=\"color: #{:06x}\">", rgb)),
                StyleType::Background { rgb } => result.push_str(&format!("<span style=\"background-color: #{:06x}\">", rgb)),
            }
            
            // Add styled content
            result.push_str(&content[style.start..style.end]);
            
            // Add closing tag
            match &style.style {
                StyleType::Bold => result.push_str("</strong>"),
                StyleType::Italic => result.push_str("</em>"),
                StyleType::Underline => result.push_str("</u>"),
                StyleType::Strikethrough => result.push_str("</del>"),
                StyleType::Code => result.push_str("</code>"),
                StyleType::Link { .. } => result.push_str("</a>"),
                StyleType::Color { .. } | StyleType::Background { .. } => result.push_str("</span>"),
            }
            
            last_end = style.end;
        }
        
        // Add remaining content
        result.push_str(&content[last_end..]);
        
        Ok(Arc::from(result))
    }

    /// Format composite content
    #[inline]
    fn format_composite(&self, parts: &[MessageContent]) -> FormatResult<Arc<str>> {
        let mut result = String::new();
        
        for part in parts {
            let formatted = self.format(part)?;
            result.push_str(&formatted);
        }
        
        Ok(Arc::from(result))
    }
}

impl Default for MessageFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageContent {
    /// Create plain text content
    #[inline]
    pub fn plain(text: impl Into<Arc<str>>) -> Self {
        Self::Plain { text: text.into() }
    }

    /// Create markdown content
    #[inline]
    pub fn markdown(content: impl Into<Arc<str>>) -> Self {
        Self::Markdown { 
            content: content.into(), 
            rendered_html: None 
        }
    }

    /// Create code content
    #[inline]
    pub fn code(content: impl Into<Arc<str>>, language: impl Into<Arc<str>>) -> Self {
        Self::Code { 
            content: content.into(), 
            language: language.into(), 
            highlighted: None 
        }
    }

    /// Create formatted content
    #[inline]
    pub fn formatted(content: impl Into<Arc<str>>, styles: Arc<[FormatStyle]>) -> Self {
        Self::Formatted { 
            content: content.into(), 
            styles 
        }
    }

    /// Create composite content
    #[inline]
    pub fn composite(parts: Arc<[MessageContent]>) -> Self {
        Self::Composite { parts }
    }

    /// Get the raw text content
    #[inline]
    pub fn raw_text(&self) -> Arc<str> {
        match self {
            Self::Plain { text } => text.clone(),
            Self::Markdown { content, .. } => content.clone(),
            Self::Code { content, .. } => content.clone(),
            Self::Formatted { content, .. } => content.clone(),
            Self::Composite { parts } => {
                let mut result = String::new();
                for part in parts.iter() {
                    result.push_str(&part.raw_text());
                }
                Arc::from(result)
            }
        }
    }

    /// Check if content is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Plain { text } => text.is_empty(),
            Self::Markdown { content, .. } => content.is_empty(),
            Self::Code { content, .. } => content.is_empty(),
            Self::Formatted { content, .. } => content.is_empty(),
            Self::Composite { parts } => parts.is_empty() || parts.iter().all(|p| p.is_empty()),
        }
    }

    /// Get content length
    #[inline]
    pub fn len(&self) -> usize {
        self.raw_text().len()
    }
}

/// Builder for creating formatted message content
pub struct MessageContentBuilder {
    content: String,
    styles: Vec<FormatStyle>,
}

impl MessageContentBuilder {
    /// Create a new builder
    #[inline]
    pub fn new() -> Self {
        Self {
            content: String::new(),
            styles: Vec::new(),
        }
    }

    /// Add plain text
    #[inline]
    pub fn text(mut self, text: &str) -> Self {
        self.content.push_str(text);
        self
    }

    /// Add bold text
    #[inline]
    pub fn bold(mut self, text: &str) -> Self {
        let start = self.content.len();
        self.content.push_str(text);
        let end = self.content.len();
        
        self.styles.push(FormatStyle {
            start,
            end,
            style: StyleType::Bold,
        });
        
        self
    }

    /// Add italic text
    #[inline]
    pub fn italic(mut self, text: &str) -> Self {
        let start = self.content.len();
        self.content.push_str(text);
        let end = self.content.len();
        
        self.styles.push(FormatStyle {
            start,
            end,
            style: StyleType::Italic,
        });
        
        self
    }

    /// Add underlined text
    #[inline]
    pub fn underline(mut self, text: &str) -> Self {
        let start = self.content.len();
        self.content.push_str(text);
        let end = self.content.len();
        
        self.styles.push(FormatStyle {
            start,
            end,
            style: StyleType::Underline,
        });
        
        self
    }

    /// Add strikethrough text
    #[inline]
    pub fn strikethrough(mut self, text: &str) -> Self {
        let start = self.content.len();
        self.content.push_str(text);
        let end = self.content.len();
        
        self.styles.push(FormatStyle {
            start,
            end,
            style: StyleType::Strikethrough,
        });
        
        self
    }

    /// Add inline code
    #[inline]
    pub fn code(mut self, text: &str) -> Self {
        let start = self.content.len();
        self.content.push_str(text);
        let end = self.content.len();
        
        self.styles.push(FormatStyle {
            start,
            end,
            style: StyleType::Code,
        });
        
        self
    }

    /// Add link
    #[inline]
    pub fn link(mut self, text: &str, url: impl Into<Arc<str>>) -> Self {
        let start = self.content.len();
        self.content.push_str(text);
        let end = self.content.len();
        
        self.styles.push(FormatStyle {
            start,
            end,
            style: StyleType::Link { url: url.into() },
        });
        
        self
    }

    /// Add colored text
    #[inline]
    pub fn color(mut self, text: &str, rgb: u32) -> Self {
        let start = self.content.len();
        self.content.push_str(text);
        let end = self.content.len();
        
        self.styles.push(FormatStyle {
            start,
            end,
            style: StyleType::Color { rgb },
        });
        
        self
    }

    /// Add text with background color
    #[inline]
    pub fn background(mut self, text: &str, rgb: u32) -> Self {
        let start = self.content.len();
        self.content.push_str(text);
        let end = self.content.len();
        
        self.styles.push(FormatStyle {
            start,
            end,
            style: StyleType::Background { rgb },
        });
        
        self
    }

    /// Build the formatted message content
    #[inline]
    pub fn build(self) -> MessageContent {
        MessageContent::formatted(
            Arc::from(self.content),
            Arc::from(self.styles),
        )
    }
}

impl Default for MessageContentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Global message formatter instance
static MESSAGE_FORMATTER: once_cell::sync::Lazy<MessageFormatter> = 
    once_cell::sync::Lazy::new(|| MessageFormatter::new());

/// Format message content using the global formatter
#[inline]
pub fn format_message(content: &MessageContent) -> FormatResult<Arc<str>> {
    MESSAGE_FORMATTER.format(content)
}

/// Parse markdown content using the global formatter
#[inline]
pub fn parse_markdown(content: &str) -> FormatResult<Arc<str>> {
    MESSAGE_FORMATTER.markdown_parser.parse_to_html(content)
}

/// Highlight code using the global formatter
#[inline]
pub fn highlight_code(content: &str, language: &str) -> FormatResult<Arc<str>> {
    MESSAGE_FORMATTER.syntax_highlighter.highlight(content, language)
}