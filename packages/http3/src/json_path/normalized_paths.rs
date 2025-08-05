//! RFC 9535 Normalized Paths Implementation (Section 2.7)
//!
//! A Normalized Path is a JSONPath expression that uniquely identifies
//! a single node in a JSON value using a canonical syntax:
//! - Use bracket notation exclusively
//! - Use single quotes for member names  
//! - Use decimal integers for array indices (no leading zeros except for 0)
//! - No whitespace except where required for parsing

use std::fmt;

use crate::json_path::{
    ast::JsonSelector,
    error::{JsonPathResult, invalid_expression_error},
};

/// A normalized JSONPath expression that uniquely identifies a single node
///
/// Normalized paths use a canonical syntax as defined in RFC 9535 Section 2.7.
/// They are used for reliable node identification and path comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormalizedPath {
    /// The canonical path segments in normalized form
    segments: Vec<PathSegment>,
    /// The complete normalized path string
    normalized_string: String,
}

/// Individual segment in a normalized path
///
/// Each segment represents one level of navigation through the JSON structure
/// using the canonical bracket notation syntax.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PathSegment {
    /// Root segment ($)
    Root,
    /// Object member access (['member_name'])
    Member(String),
    /// Array index access ([index])
    Index(i64),
}

/// Normalized Path Generator and Validator
pub struct NormalizedPathProcessor;

impl NormalizedPathProcessor {
    /// Generate normalized path from JSONPath selectors
    ///
    /// Takes a sequence of JSONPath selectors and converts them to
    /// a normalized path if they represent a unique single-node path.
    ///
    /// # Errors
    ///
    /// Returns error if the selectors don't represent a normalized path:
    /// - Contains wildcards, filters, or other multi-node selectors
    /// - Contains recursive descent that doesn't target a specific node
    /// - Contains union selectors
    /// - Invalid or malformed selectors
    #[inline]
    pub fn generate_normalized_path(
        selectors: &[JsonSelector]
    ) -> JsonPathResult<NormalizedPath> {
        let mut segments = Vec::new();
        
        // First selector should always be Root for normalized paths
        if selectors.is_empty() {
            return Ok(NormalizedPath::root());
        }

        // Validate and convert each selector
        for (index, selector) in selectors.iter().enumerate() {
            match selector {
                JsonSelector::Root => {
                    if index != 0 {
                        return Err(invalid_expression_error(
                            "",
                            "root selector can only appear at the beginning",
                            Some(index),
                        ));
                    }
                    segments.push(PathSegment::Root);
                }
                
                JsonSelector::Child { name, .. } => {
                    Self::validate_member_name(name)?;
                    segments.push(PathSegment::Member(name.clone()));
                }
                
                JsonSelector::Index { index, from_end } => {
                    if *from_end {
                        return Err(invalid_expression_error(
                            "",
                            "normalized paths cannot contain negative indices",
                            Some(index.wrapping_abs() as usize),
                        ));
                    }
                    if *index < 0 {
                        return Err(invalid_expression_error(
                            "",
                            "normalized paths require non-negative array indices",
                            Some((*index).wrapping_abs() as usize),
                        ));
                    }
                    segments.push(PathSegment::Index(*index));
                }
                
                // These selectors cannot appear in normalized paths
                JsonSelector::Wildcard => {
                    return Err(invalid_expression_error(
                        "",
                        "normalized paths cannot contain wildcard selectors",
                        Some(index),
                    ));
                }
                JsonSelector::Slice { .. } => {
                    return Err(invalid_expression_error(
                        "",
                        "normalized paths cannot contain slice selectors",
                        Some(index),
                    ));
                }
                JsonSelector::Filter { .. } => {
                    return Err(invalid_expression_error(
                        "",
                        "normalized paths cannot contain filter selectors",
                        Some(index),
                    ));
                }
                JsonSelector::Union { .. } => {
                    return Err(invalid_expression_error(
                        "",
                        "normalized paths cannot contain union selectors",
                        Some(index),
                    ));
                }
                JsonSelector::RecursiveDescent => {
                    return Err(invalid_expression_error(
                        "",
                        "normalized paths cannot contain recursive descent",
                        Some(index),
                    ));
                }
            }
        }

        // Generate the normalized string representation
        let normalized_string = Self::segments_to_string(&segments);
        
        Ok(NormalizedPath {
            segments,
            normalized_string,
        })
    }

    /// Parse a normalized path string into segments
    ///
    /// Validates that the input string conforms to normalized path syntax
    /// and converts it to internal representation.
    #[inline]
    pub fn parse_normalized_path(path: &str) -> JsonPathResult<NormalizedPath> {
        if path == "$" {
            return Ok(NormalizedPath::root());
        }

        if !path.starts_with('$') {
            return Err(invalid_expression_error(
                path,
                "normalized paths must start with $",
                Some(0),
            ));
        }

        let mut segments = vec![PathSegment::Root];
        let remaining = &path[1..];
        
        if remaining.is_empty() {
            return Ok(NormalizedPath {
                segments,
                normalized_string: path.to_string(),
            });
        }

        let mut chars = remaining.chars().peekable();
        let mut position = 1; // Start after $

        while chars.peek().is_some() {
            if chars.next() != Some('[') {
                return Err(invalid_expression_error(
                    path,
                    "normalized paths must use bracket notation",
                    Some(position),
                ));
            }
            position += 1;

            // Parse the bracket content
            let segment = Self::parse_bracket_content(&mut chars, &mut position, path)?;
            segments.push(segment);

            if chars.next() != Some(']') {
                return Err(invalid_expression_error(
                    path,
                    "expected closing bracket",
                    Some(position),
                ));
            }
            position += 1;
        }

        Ok(NormalizedPath {
            segments,
            normalized_string: path.to_string(),
        })
    }

    /// Parse content within brackets
    #[inline]
    fn parse_bracket_content(
        chars: &mut std::iter::Peekable<std::str::Chars>,
        position: &mut usize,
        full_path: &str,
    ) -> JsonPathResult<PathSegment> {
        let start_pos = *position;
        
        match chars.peek() {
            Some('\'') => {
                // Single-quoted string (member name)
                chars.next(); // consume opening quote
                *position += 1;
                
                let mut member_name = String::new();
                
                while let Some(ch) = chars.next() {
                    *position += 1;
                    
                    if ch == '\'' {
                        // End of string
                        Self::validate_member_name(&member_name)?;
                        return Ok(PathSegment::Member(member_name));
                    } else if ch == '\\' {
                        // Escape sequence
                        match chars.next() {
                            Some('\'') => {
                                member_name.push('\'');
                                *position += 1;
                            }
                            Some('\\') => {
                                member_name.push('\\');
                                *position += 1;
                            }
                            Some(escaped) => {
                                return Err(invalid_expression_error(
                                    full_path,
                                    &format!("invalid escape sequence \\{}", escaped),
                                    Some(*position),
                                ));
                            }
                            None => {
                                return Err(invalid_expression_error(
                                    full_path,
                                    "unterminated escape sequence",
                                    Some(*position),
                                ));
                            }
                        }
                    } else {
                        member_name.push(ch);
                    }
                }
                
                Err(invalid_expression_error(
                    full_path,
                    "unterminated string literal",
                    Some(start_pos),
                ))
            }
            
            Some(ch) if ch.is_ascii_digit() => {
                // Array index
                let mut index_str = String::new();
                
                while let Some(&ch) = chars.peek() {
                    if ch.is_ascii_digit() {
                        index_str.push(ch);
                        chars.next();
                        *position += 1;
                    } else {
                        break;
                    }
                }
                
                // Validate no leading zeros (except for "0")
                if index_str.len() > 1 && index_str.starts_with('0') {
                    return Err(invalid_expression_error(
                        full_path,
                        "array indices cannot have leading zeros",
                        Some(start_pos),
                    ));
                }
                
                let index = index_str.parse::<i64>()
                    .map_err(|_| invalid_expression_error(
                        full_path,
                        "invalid array index",
                        Some(start_pos),
                    ))?;
                
                if index < 0 {
                    return Err(invalid_expression_error(
                        full_path,
                        "normalized paths require non-negative array indices",
                        Some(start_pos),
                    ));
                }
                
                Ok(PathSegment::Index(index))
            }
            
            _ => Err(invalid_expression_error(
                full_path,
                "expected string literal or array index",
                Some(start_pos),
            )),
        }
    }

    /// Validate member name according to normalized path rules
    #[inline]
    fn validate_member_name(name: &str) -> JsonPathResult<()> {
        // Member names in normalized paths must be valid UTF-8 strings
        // No additional restrictions beyond basic JSON string requirements
        if name.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
            return Err(invalid_expression_error(
                "",
                "member names cannot contain control characters",
                None,
            ));
        }
        Ok(())
    }

    /// Convert segments to normalized string representation
    #[inline]
    fn segments_to_string(segments: &[PathSegment]) -> String {
        let mut result = String::new();
        
        for segment in segments {
            match segment {
                PathSegment::Root => result.push('$'),
                PathSegment::Member(name) => {
                    result.push('[');
                    result.push('\'');
                    // Escape quotes and backslashes in member names
                    for ch in name.chars() {
                        if ch == '\'' || ch == '\\' {
                            result.push('\\');
                        }
                        result.push(ch);
                    }
                    result.push('\'');
                    result.push(']');
                }
                PathSegment::Index(index) => {
                    result.push('[');
                    result.push_str(&index.to_string());
                    result.push(']');
                }
            }
        }
        
        result
    }
}

impl NormalizedPath {
    /// Create a root normalized path ($)
    #[inline]
    pub fn root() -> Self {
        Self {
            segments: vec![PathSegment::Root],
            normalized_string: "$".to_string(),
        }
    }

    /// Get the canonical string representation
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.normalized_string
    }

    /// Get the path segments
    #[inline]
    pub fn segments(&self) -> &[PathSegment] {
        &self.segments
    }

    /// Check if this is the root path ($)
    #[inline]
    pub fn is_root(&self) -> bool {
        self.segments.len() == 1 && matches!(self.segments[0], PathSegment::Root)
    }

    /// Get the depth of this path (number of non-root segments)
    #[inline]
    pub fn depth(&self) -> usize {
        self.segments.len().saturating_sub(1)
    }

    /// Create a child path by appending a member access
    #[inline]
    pub fn child_member(&self, member_name: &str) -> JsonPathResult<Self> {
        NormalizedPathProcessor::validate_member_name(member_name)?;
        
        let mut new_segments = self.segments.clone();
        new_segments.push(PathSegment::Member(member_name.to_string()));
        
        let normalized_string = NormalizedPathProcessor::segments_to_string(&new_segments);
        
        Ok(Self {
            segments: new_segments,
            normalized_string,
        })
    }

    /// Create a child path by appending an array index access
    #[inline]
    pub fn child_index(&self, index: i64) -> JsonPathResult<Self> {
        if index < 0 {
            return Err(invalid_expression_error(
                "",
                "normalized paths require non-negative array indices",
                None,
            ));
        }
        
        let mut new_segments = self.segments.clone();
        new_segments.push(PathSegment::Index(index));
        
        let normalized_string = NormalizedPathProcessor::segments_to_string(&new_segments);
        
        Ok(Self {
            segments: new_segments,
            normalized_string,
        })
    }

    /// Get the parent path (all segments except the last)
    #[inline]
    pub fn parent(&self) -> Option<Self> {
        if self.segments.len() <= 1 {
            return None; // Root has no parent
        }
        
        let parent_segments = self.segments[..self.segments.len() - 1].to_vec();
        let normalized_string = NormalizedPathProcessor::segments_to_string(&parent_segments);
        
        Some(Self {
            segments: parent_segments,
            normalized_string,
        })
    }

    /// Check if this path is a descendant of another path
    #[inline]
    pub fn is_descendant_of(&self, ancestor: &NormalizedPath) -> bool {
        if self.segments.len() <= ancestor.segments.len() {
            return false;
        }
        
        self.segments[..ancestor.segments.len()] == ancestor.segments
    }

    /// Check if this path is an ancestor of another path
    #[inline]
    pub fn is_ancestor_of(&self, descendant: &NormalizedPath) -> bool {
        descendant.is_descendant_of(self)
    }
}

impl fmt::Display for NormalizedPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.normalized_string)
    }
}

impl fmt::Display for PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathSegment::Root => write!(f, "$"),
            PathSegment::Member(name) => write!(f, "['{}']", name),
            PathSegment::Index(index) => write!(f, "[{}]", index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_path() {
        let root = NormalizedPath::root();
        assert_eq!(root.as_str(), "$");
        assert!(root.is_root());
        assert_eq!(root.depth(), 0);
        assert!(root.parent().is_none());
    }

    #[test]
    fn test_member_path() {
        let root = NormalizedPath::root();
        let member_path = root.child_member("store")
            .expect("Failed to create child member path for 'store'");
        assert_eq!(member_path.as_str(), "$['store']");
        assert!(!member_path.is_root());
        assert_eq!(member_path.depth(), 1);
        assert_eq!(member_path.parent()
            .expect("Failed to get parent of member path").as_str(), "$");
    }

    #[test]
    fn test_index_path() {
        let root = NormalizedPath::root();
        let member_path = root.child_member("items")
            .expect("Failed to create child member path for 'items'");
        let index_path = member_path.child_index(0)
            .expect("Failed to create child index path for index 0");
        assert_eq!(index_path.as_str(), "$['items'][0]");
        assert_eq!(index_path.depth(), 2);
    }

    #[test]
    fn test_complex_path() {
        let root = NormalizedPath::root();
        let complex = root
            .child_member("store")
            .expect("Failed to create child member path for 'store'")
            .child_member("book")
            .expect("Failed to create child member path for 'book'")
            .child_index(0)
            .expect("Failed to create child index path for index 0")
            .child_member("title")
            .expect("Failed to create child member path for 'title'");
        
        assert_eq!(complex.as_str(), "$['store']['book'][0]['title']");
        assert_eq!(complex.depth(), 4);
    }

    #[test]
    fn test_parse_normalized_path() {
        let parsed = NormalizedPathProcessor::parse_normalized_path("$['store']['book'][0]['title']")
            .expect("Failed to parse normalized path expression");
        assert_eq!(parsed.as_str(), "$['store']['book'][0]['title']");
        assert_eq!(parsed.depth(), 4);
    }

    #[test]
    fn test_path_relationships() {
        let parent = NormalizedPath::root().child_member("store")
            .expect("Failed to create child member path for 'store'");
        let child = parent.child_member("book")
            .expect("Failed to create child member path for 'book'");
        
        assert!(child.is_descendant_of(&parent));
        assert!(parent.is_ancestor_of(&child));
        assert!(!parent.is_descendant_of(&child));
    }

    #[test]
    fn test_invalid_paths() {
        // Negative index
        assert!(NormalizedPath::root().child_index(-1).is_err());
        
        // Invalid parse - no bracket notation
        assert!(NormalizedPathProcessor::parse_normalized_path("$.store").is_err());
        
        // Leading zeros
        assert!(NormalizedPathProcessor::parse_normalized_path("$[01]").is_err());
    }
}