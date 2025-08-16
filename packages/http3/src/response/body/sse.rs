//! Server-Sent Events (SSE) parsing and processing functionality
//!
//! Handles SSE parsing according to the Server-Sent Events specification
//! with support for multi-line data fields, event types, IDs, and retry directives.

use crate::response::core::HttpResponse;

impl HttpResponse {
    /// Get Server-Sent Events - returns Vec<SseEvent> directly
    ///
    /// Get SSE events if this is an SSE response
    /// Returns empty `Vec` if not an SSE response
    #[must_use]
    pub fn sse(&self) -> Vec<SseEvent> {
        let body = String::from_utf8_lossy(self.body());
        Self::parse_sse_events(&body)
    }

    /// Parse SSE events according to the Server-Sent Events specification
    /// Handles multi-line data fields, event types, IDs, and retry directives
    fn parse_sse_events(body: &str) -> Vec<SseEvent> {
        let mut events = Vec::new();
        let mut current_event = SseEvent {
            data: None,
            event_type: None,
            id: None,
            retry: None,
        };
        let mut data_lines = Vec::new();

        for line in body.lines() {
            let line = line.trim_end_matches('\r'); // Handle CRLF endings

            // Empty line indicates end of event
            if line.is_empty() {
                if !data_lines.is_empty()
                    || current_event.event_type.is_some()
                    || current_event.id.is_some()
                    || current_event.retry.is_some()
                {
                    // Join data lines with newlines (SSE spec requirement)
                    if !data_lines.is_empty() {
                        current_event.data = Some(data_lines.join("\n"));
                    }

                    events.push(current_event);

                    // Reset for next event
                    current_event = SseEvent {
                        data: None,
                        event_type: None,
                        id: None,
                        retry: None,
                    };
                    data_lines.clear();
                }
                continue;
            }

            // Skip comment lines (start with :)
            if line.starts_with(':') {
                continue;
            }

            // Parse field: value pairs
            if let Some(colon_pos) = line.find(':') {
                let field = &line[..colon_pos];
                let value = line[colon_pos + 1..].trim_start_matches(' ');

                match field {
                    "data" => {
                        data_lines.push(value.to_string());
                    }
                    "event" => {
                        current_event.event_type = Some(value.to_string());
                    }
                    "id" => {
                        // ID field must not contain null characters (spec requirement)
                        if !value.contains('\0') {
                            current_event.id = Some(value.to_string());
                        }
                    }
                    "retry" => {
                        // retry field must be a valid number (milliseconds)
                        let retry_ms = value.parse::<u64>().ok();
                        if let Some(retry_ms) = retry_ms {
                            current_event.retry = Some(retry_ms);
                        }
                    }
                    _ => {
                        // Ignore unknown fields (spec allows this)
                    }
                }
            } else {
                // Line without colon is treated as "data: <line>"
                data_lines.push(line.to_string());
            }
        }

        // Handle final event if stream doesn't end with empty line
        if !data_lines.is_empty()
            || current_event.event_type.is_some()
            || current_event.id.is_some()
            || current_event.retry.is_some()
        {
            if !data_lines.is_empty() {
                current_event.data = Some(data_lines.join("\n"));
            }
            events.push(current_event);
        }

        events
    }
}
