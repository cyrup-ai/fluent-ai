//! Special Token Management
//!
//! Provides comprehensive special token handling including BOS, EOS, PAD, UNK,
//! CLS, SEP, and MASK tokens with efficient lookup and validation.

// Removed unused import: HashMap

use super::core::CandleTokenizer;

impl CandleTokenizer {
    /// Get token ID for specific token string
    #[inline(always)]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner().token_to_id(token)
    }

    /// Get token string for specific token ID
    #[inline(always)]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner().id_to_token(id)
    }

    /// Get special token ID by type
    pub fn get_special_token_id(&self, token_type: &str) -> Option<u32> {
        let special_tokens = self.special_tokens();
        match token_type.to_lowercase().as_str() {
            "bos" | "start" => special_tokens
                .get("<s>")
                .or_else(|| special_tokens.get("[BOS]"))
                .or_else(|| special_tokens.get("<|startoftext|>"))
                .copied(),
            "eos" | "end" => special_tokens
                .get("</s>")
                .or_else(|| special_tokens.get("[EOS]"))
                .or_else(|| special_tokens.get("<|endoftext|>"))
                .copied(),
            "pad" | "padding" => special_tokens
                .get("<pad>")
                .or_else(|| special_tokens.get("[PAD]"))
                .copied(),
            "unk" | "unknown" => special_tokens
                .get("<unk>")
                .or_else(|| special_tokens.get("[UNK]"))
                .copied(),
            "cls" | "class" => special_tokens
                .get("<cls>")
                .or_else(|| special_tokens.get("[CLS]"))
                .copied(),
            "sep" | "separator" => special_tokens
                .get("<sep>")
                .or_else(|| special_tokens.get("[SEP]"))
                .copied(),
            "mask" => special_tokens
                .get("<mask>")
                .or_else(|| special_tokens.get("[MASK]"))
                .copied(),
            _ => special_tokens.get(token_type).copied()}
    }

    /// Get EOS (end-of-sequence) token ID
    #[inline(always)]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.get_special_token_id("eos")
    }

    /// Check if token ID is a special token
    pub fn is_special_token(&self, token_id: u32) -> bool {
        self.special_tokens().values().any(|&id| id == token_id)
    }

    /// Apply chat template if supported by tokenizer
    pub fn apply_chat_template(
        &self,
        messages: &[crate::types::CandleMessage],
        add_generation_prompt: bool,
    ) -> crate::error::CandleResult<String> {
        // This would use the tokenizer's chat template functionality
        // For now, implement a basic chat format
        let mut formatted = String::new();

        for message in messages {
            match message.role.as_str() {
                "system" => formatted.push_str(&format!("<|system|>\n{}\n", message.content)),
                "user" => formatted.push_str(&format!("<|user|>\n{}\n", message.content)),
                "assistant" => formatted.push_str(&format!("<|assistant|>\n{}\n", message.content)),
                _ => formatted.push_str(&format!("<|{}|>\n{}\n", message.role, message.content))}
        }

        if add_generation_prompt {
            formatted.push_str("<|assistant|>\n");
        }

        Ok(formatted)
    }
}