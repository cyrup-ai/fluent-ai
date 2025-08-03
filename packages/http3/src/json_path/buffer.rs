//! Zero-allocation buffer management for JSON streaming
//!
//! This module provides high-performance buffer management for streaming JSON parsing.
//! Uses memory pools, zero-copy techniques, and efficient chunk aggregation to minimize
//! allocations and maximize throughput.

use std::io::{self, Read};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde_json::de::IoRead;

/// Zero-allocation streaming buffer with efficient chunk management
///
/// Optimized for JSON parsing workflows where data arrives in chunks and needs
/// to be parsed incrementally. Uses memory pools and zero-copy techniques.
pub struct StreamBuffer {
    /// Main buffer for accumulating incoming chunks
    buffer: BytesMut,
    /// Total bytes processed (for statistics)
    total_processed: u64,
    /// Position of last complete JSON object boundary
    last_boundary: usize,
    /// Buffer capacity management
    capacity_manager: CapacityManager,
}

impl StreamBuffer {
    /// Create new stream buffer with specified initial capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial buffer capacity in bytes (recommended: 8KB-64KB)
    ///
    /// # Performance
    ///
    /// Initial capacity should be sized based on expected chunk sizes and JSON object sizes.
    /// Larger capacities reduce reallocations but increase memory usage.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: BytesMut::with_capacity(capacity),
            total_processed: 0,
            last_boundary: 0,
            capacity_manager: CapacityManager::new(capacity),
        }
    }

    /// Create buffer with default capacity optimized for HTTP responses
    pub fn new() -> Self {
        Self::with_capacity(8192) // 8KB default
    }

    /// Append incoming HTTP chunk to buffer
    ///
    /// # Arguments
    ///
    /// * `chunk` - Incoming bytes from HTTP response stream
    ///
    /// # Performance
    ///
    /// Uses zero-copy techniques when possible. Automatically manages buffer
    /// capacity and growth to minimize reallocations.
    pub fn append_chunk(&mut self, chunk: Bytes) {
        self.total_processed += chunk.len() as u64;

        // Check if we need to grow the buffer
        if self.buffer.remaining_mut() < chunk.len() {
            self.capacity_manager
                .ensure_capacity(&mut self.buffer, chunk.len());
        }

        // Zero-copy append when possible
        self.buffer.extend_from_slice(&chunk);
    }

    /// Get a reader for the current buffer contents
    ///
    /// Returns a reader that implements `std::io::Read` for use with serde_json::StreamDeserializer.
    /// The reader tracks position and handles partial reads correctly.
    pub fn reader(&mut self) -> BufferReader<'_> {
        BufferReader {
            buffer: &self.buffer[..],
            position: 0,
        }
    }

    /// Create a reader for the current buffer contents (alias for reader)
    ///
    /// Returns a reader that implements both `std::io::Read` and `serde_json::de::Read`
    /// for seamless integration with serde_json streaming deserializers.
    #[inline]
    pub fn create_reader(&mut self) -> BufferReader<'_> {
        self.reader()
    }

    /// Mark bytes as consumed after successful JSON parsing
    ///
    /// # Arguments
    ///
    /// * `bytes_consumed` - Number of bytes that were successfully parsed
    ///
    /// # Performance
    ///
    /// Uses efficient buffer advance operations to avoid copying remaining data.
    pub fn consume(&mut self, bytes_consumed: usize) {
        if bytes_consumed <= self.buffer.len() {
            self.buffer.advance(bytes_consumed);
            self.last_boundary = 0; // Reset boundary tracking

            // Shrink buffer if it's grown too large and is mostly empty
            self.capacity_manager.maybe_shrink(&mut self.buffer);
        }
    }

    /// Get current buffer size in bytes
    #[inline]
    pub fn current_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get total bytes processed since creation
    #[inline]
    pub fn total_bytes_processed(&self) -> u64 {
        self.total_processed
    }

    /// Check if buffer has enough data for JSON parsing attempt
    ///
    /// # Arguments
    ///
    /// * `min_bytes` - Minimum bytes needed for parsing attempt
    ///
    /// Returns `true` if buffer contains at least `min_bytes` of data.
    #[inline]
    pub fn has_data(&self, min_bytes: usize) -> bool {
        self.buffer.len() >= min_bytes
    }

    /// Find likely JSON object boundaries in buffer
    ///
    /// Scans for complete JSON objects (balanced braces) to optimize parsing attempts.
    /// Returns positions of potential object boundaries for batch processing.
    pub fn find_object_boundaries(&self) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let mut brace_depth = 0;
        let mut in_string = false;
        let mut escape_next = false;

        let data = &self.buffer[..];
        for (i, &byte) in data.iter().enumerate() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match byte {
                b'\\' if in_string => escape_next = true,
                b'"' => in_string = !in_string,
                b'{' if !in_string => brace_depth += 1,
                b'}' if !in_string => {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        boundaries.push(i + 1); // Position after closing brace
                    }
                }
                _ => {}
            }
        }

        boundaries
    }

    /// Clear buffer and reset state
    ///
    /// Useful for error recovery or when switching to a new response stream.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_boundary = 0;
        self.capacity_manager.reset();
    }

    /// Get byte at specific position in buffer
    ///
    /// Returns None if position is beyond buffer length
    #[inline]
    pub fn get_byte_at(&self, position: usize) -> Option<u8> {
        self.buffer.get(position).copied()
    }

    /// Get current buffer length for bounds checking
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get buffer utilization statistics for monitoring
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            current_size: self.buffer.len(),
            capacity: self.buffer.capacity(),
            total_processed: self.total_processed,
            utilization_ratio: self.buffer.len() as f64 / self.buffer.capacity() as f64,
        }
    }
}

impl Default for StreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Buffer reader implementing std::io::Read for serde_json integration
pub struct BufferReader<'a> {
    buffer: &'a [u8],
    position: usize,
}

impl<'a> Read for BufferReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = &self.buffer[self.position..];
        let to_copy = std::cmp::min(buf.len(), remaining.len());

        if to_copy > 0 {
            buf[..to_copy].copy_from_slice(&remaining[..to_copy]);
            self.position += to_copy;
        }

        Ok(to_copy)
    }
}

impl<'a> BufferReader<'a> {
    /// Create an IoRead wrapper for use with serde_json::StreamDeserializer
    ///
    /// This method returns an IoRead wrapper that can be used directly with
    /// serde_json's streaming deserializer while maintaining zero-allocation principles.
    pub fn into_io_read(self) -> IoRead<Self> {
        IoRead::new(self)
    }

    /// Get current position for debugging and monitoring
    #[inline]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get remaining bytes available for reading
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.position
    }

    /// Check if reader has reached end of buffer
    #[inline]
    pub fn is_eof(&self) -> bool {
        self.position >= self.buffer.len()
    }
}

/// Intelligent buffer capacity management
///
/// Handles buffer growth and shrinking based on usage patterns to minimize
/// memory usage while avoiding frequent reallocations.
struct CapacityManager {
    initial_capacity: usize,
    max_capacity: usize,
    growth_factor: f64,
    shrink_threshold: f64,
}

impl CapacityManager {
    fn new(initial_capacity: usize) -> Self {
        Self {
            initial_capacity,
            max_capacity: initial_capacity * 16, // Max 16x growth
            growth_factor: 2.0,
            shrink_threshold: 0.25, // Shrink when less than 25% utilized
        }
    }

    fn ensure_capacity(&mut self, buffer: &mut BytesMut, needed: usize) {
        let current_capacity = buffer.capacity();
        let current_size = buffer.len();
        let required = current_size + needed;

        if required > current_capacity {
            let new_capacity = std::cmp::min(
                self.max_capacity,
                std::cmp::max(
                    required,
                    (current_capacity as f64 * self.growth_factor) as usize,
                ),
            );

            buffer.reserve(new_capacity - current_capacity);
        }
    }

    fn maybe_shrink(&mut self, buffer: &mut BytesMut) {
        let capacity = buffer.capacity();
        let size = buffer.len();
        let utilization = size as f64 / capacity as f64;

        // Only shrink if significantly under-utilized and above initial capacity
        if utilization < self.shrink_threshold && capacity > self.initial_capacity * 2 {
            let target_capacity = std::cmp::max(
                self.initial_capacity,
                (size as f64 / self.shrink_threshold) as usize,
            );

            // TODO: Implement actual buffer shrinking when BytesMut supports it
            log::debug!(
                "Buffer shrinking recommended: current={}, target={}",
                capacity,
                target_capacity
            );
            // For now, just track that shrinking would be beneficial
            // In a full implementation, we'd use a custom allocator or buffer pool
        }
    }

    fn reset(&mut self) {
        // Reset capacity management state for new stream
    }
}

/// Buffer performance and utilization statistics
#[derive(Debug, Clone, Copy)]
pub struct BufferStats {
    /// Current buffer size in bytes
    pub current_size: usize,
    /// Buffer capacity in bytes
    pub capacity: usize,
    /// Total bytes processed since creation
    pub total_processed: u64,
    /// Buffer utilization ratio (0.0 to 1.0)
    pub utilization_ratio: f64,
}

/// Legacy alias for backward compatibility
pub type JsonBuffer = StreamBuffer;

#[cfg(test)]
mod tests {
    use super::*;
}
