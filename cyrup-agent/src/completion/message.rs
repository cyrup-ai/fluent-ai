//! Zero-alloc fluent builder for [`Message`] values.
//!
//! * Two dedicated entry points: `MessageBuilder::user()` and
//!   `MessageBuilder::assistant()`.
//! * Compile-time guarantee that **at least one piece of content** is present
//!   before `.build()` becomes available (const-generic typestate).
//! * Absolutely no Arc / Mutex / heap-allocs on the hot-path.
//!
//! # Example
//! ```rust
//! use rig::completion::message::{MessageBuilder, ImageDetail};
//!
//! let msg = MessageBuilder::user()
//!     .text("An image please")
//!     .image(
//!         "data:…base64…",
//!         None,
//!         Some(ImageMediaType::PNG),
//!         Some(ImageDetail::High),
//!     )
//!     .build();
//! ```
#![allow(clippy::type_complexity)]

use super::*;
use crate::OneOrMany;

/// Internal role marker.
#[derive(Clone, Copy)]
enum Role {
    User,
    Assistant,
}

/// Builder with const-generic flag that tracks whether we already have content.
///
/// * `HAS_CONTENT = false`  →  `.build()` **not** available yet
/// * `HAS_CONTENT = true`   →  `.build()` **is** available
pub struct MessageBuilder<const HAS_CONTENT: bool> {
    role: Role,
    // We hold the first item separately so the happy path (`OneOrMany::one`)
    // never allocates.  Additional items (rare) spill into the Vec inside
    // `OneOrMany::many` – still zero-cost if only one item is pushed.
    first: Option<Content>,
    overflow: Vec<Content>,
}

/// Erased content so we can keep a single field pair even though
/// `UserContent` and `AssistantContent` are distinct enums.
enum Content {
    User(UserContent),
    Assistant(AssistantContent),
}

// ---------- Smart constructors ---------------------------------------------

impl MessageBuilder<false> {
    #[inline(always)]
    pub fn user() -> Self {
        Self {
            role: Role::User,
            first: None,
            overflow: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn assistant() -> Self {
        Self {
            role: Role::Assistant,
            first: None,
            overflow: Vec::new(),
        }
    }
}

// ---------- Content-adding fluent methods (return a builder with HAS_CONTENT=true) -----

impl<const HAD: bool> MessageBuilder<HAD> {
    #[inline(always)]
    pub fn text(self, txt: impl Into<String>) -> MessageBuilder<true> {
        self.push(match self.role {
            Role::User => Content::User(UserContent::Text(txt.into().into())),
            Role::Assistant => Content::Assistant(AssistantContent::Text(txt.into().into())),
        })
    }

    #[inline(always)]
    pub fn image(
        self,
        data: impl Into<String>,
        format: Option<ContentFormat>,
        ty: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> MessageBuilder<true> {
        let img = Image {
            data: data.into(),
            format,
            media_type: ty,
            detail,
        };
        self.push(Content::User(UserContent::Image(img)))
    }

    #[inline(always)]
    pub fn audio(
        self,
        data: impl Into<String>,
        format: Option<ContentFormat>,
        ty: Option<AudioMediaType>,
    ) -> MessageBuilder<true> {
        let audio = Audio {
            data: data.into(),
            format,
            media_type: ty,
        };
        self.push(Content::User(UserContent::Audio(audio)))
    }

    #[inline(always)]
    pub fn document(
        self,
        data: impl Into<String>,
        format: Option<ContentFormat>,
        ty: Option<DocumentMediaType>,
    ) -> MessageBuilder<true> {
        let doc = Document {
            data: data.into(),
            format,
            media_type: ty,
        };
        self.push(Content::User(UserContent::Document(doc)))
    }

    #[inline(always)]
    pub fn tool_call(
        self,
        id: impl Into<String>,
        name: impl Into<String>,
        args: serde_json::Value,
    ) -> MessageBuilder<true> {
        debug_assert!(matches!(self.role, Role::Assistant));
        let tc = ToolCall {
            id: id.into(),
            function: ToolFunction {
                name: name.into(),
                arguments: args,
            },
        };
        self.push(Content::Assistant(AssistantContent::ToolCall(tc)))
    }

    #[inline(always)]
    pub fn tool_result(
        self,
        id: impl Into<String>,
        content: impl Into<String>,
    ) -> MessageBuilder<true> {
        debug_assert!(matches!(self.role, Role::User));
        let tr = ToolResult {
            id: id.into(),
            content: OneOrMany::one(ToolResultContent::Text(content.into().into())),
        };
        self.push(Content::User(UserContent::ToolResult(tr)))
    }

    // --- private helper ----------------------------------------------------
    #[inline(always)]
    fn push(mut self, item: Content) -> MessageBuilder<true> {
        match &mut self.first {
            None => self.first = Some(item),
            Some(_) => self.overflow.push(item),
        }
        // SAFETY: We just added content ⇒ HAS_CONTENT=true
        unsafe { std::mem::transmute(self) }
    }
}

// ---------- .build() only available when we KNOW we have content ------------

impl MessageBuilder<true> {
    #[inline(always)]
    pub fn build(mut self) -> Message {
        // Move out the first element (we know it exists)
        let first = self.first.take().expect("typestate bug: no content");

        // Convert (first + overflow) into OneOrMany<…>
        macro_rules! finish {
            ($variant:ident, $enum_ty:ty) => {{
                let mut vec = self
                    .overflow
                    .into_iter()
                    .filter_map(|c| {
                        if let Content::$variant(x) = c {
                            Some(x)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<$enum_ty>>();

                match first {
                    Content::$variant(first_val) => {
                        if vec.is_empty() {
                            Message::$variant {
                                content: OneOrMany::one(first_val),
                            }
                        } else {
                            vec.insert(0, first_val);
                            Message::$variant {
                                content: OneOrMany::many(vec),
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }};
        }

        match self.role {
            Role::User => finish!(User, UserContent),
            Role::Assistant => finish!(Assistant, AssistantContent),
        }
    }
}
