//! Search result sorting functionality
//!
//! Contains algorithms for sorting search results by different criteria
//! including relevance, date, and user.

use std::sync::Arc;

use super::core::ChatSearchIndex;
use super::super::types::{SearchResult, SortOrder};

impl ChatSearchIndex {
    /// Sort search results by specified order
    pub(super) fn sort_results(&self, results: &mut Vec<SearchResult>, sort_order: &SortOrder) {
        match sort_order {
            SortOrder::Relevance => {
                results.sort_by(|a, b| {
                    b.relevance_score
                        .partial_cmp(&a.relevance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortOrder::DateDescending => {
                results.sort_by(|a, b| {
                    b.message
                        .message
                        .timestamp
                        .cmp(&a.message.message.timestamp)
                });
            }
            SortOrder::DateAscending => {
                results.sort_by(|a, b| {
                    a.message
                        .message
                        .timestamp
                        .cmp(&b.message.message.timestamp)
                });
            }
            SortOrder::UserAscending => {
                results.sort_by(|a, b| {
                    format!("{:?}", a.message.message.role)
                        .cmp(&format!("{:?}", b.message.message.role))
                });
            }
            SortOrder::UserDescending => {
                results.sort_by(|a, b| {
                    format!("{:?}", b.message.message.role)
                        .cmp(&format!("{:?}", a.message.message.role))
                });
            }
        }
    }
}