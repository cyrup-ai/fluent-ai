//! Concurrency primitives and utilities

use std::sync::Arc;

use fluent_ai_async::{AsyncTask, AsyncStream};
use std::sync::Mutex;
use crossbeam_channel::{bounded, unbounded};

use crate::core::ChannelError;

/// A multi-producer, single-consumer channel for sending values between tasks
pub struct Channel<T> {
    sender: crossbeam_channel::Sender<T>,
    receiver: Arc<Mutex<crossbeam_channel::Receiver<T>>>}

impl<T> Clone for Channel<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            receiver: self.receiver.clone()}
    }
}

impl<T: Send + 'static> Channel<T> {
    /// Create a new channel with the given buffer size
    pub fn new(buffer: usize) -> Self {
        let (sender, receiver) = if buffer == 0 {
            unbounded()
        } else {
            bounded(buffer)
        };
        Self {
            sender,
            receiver: Arc::new(Mutex::new(receiver))}
    }

    /// Send a value into the channel
    pub fn send(&self, value: T) -> AsyncStream<Result<(), ChannelError>> {
        let sender = self.sender.clone();
        AsyncStream::with_channel(|stream_sender| {
            std::thread::spawn(move || {
                let result = sender
                    .send(value)
                    .map_err(|_| ChannelError::SendError);
                let _ = stream_sender.send(result);
            });
        })
    }

    /// Receive the next value from the channel
    pub fn recv(&self) -> AsyncStream<Result<T, ChannelError>> {
        let receiver = self.receiver.clone();
        AsyncStream::with_channel(|stream_sender| {
            std::thread::spawn(move || {
                let result = {
                    if let Ok(guard) = receiver.try_lock() {
                        guard.recv().map_err(|_| ChannelError::Closed)
                    } else {
                        Err(ChannelError::Closed)
                    }
                };
                let _ = stream_sender.send(result);
            });
        })
    }

    /// Create a new receiver that can be used to receive values from this channel
    pub fn subscribe(&self) -> AsyncStream<Result<T, ChannelError>> {
        let receiver = self.receiver.clone();
        AsyncStream::with_channel(|stream_sender| {
            std::thread::spawn(move || {
                if let Ok(guard) = receiver.try_lock() {
                    while let Ok(value) = guard.recv() {
                        if stream_sender.send(Ok(value)).is_err() {
                            break;
                        }
                    }
                }
                let _ = stream_sender.send(Err(ChannelError::Closed));
            });
        })
    }
}

/// A oneshot channel for sending a single value between tasks
pub struct OneshotChannel<T> {
    sender: Option<crossbeam_channel::Sender<T>>,
    receiver: crossbeam_channel::Receiver<T>}

impl<T> OneshotChannel<T> {
    /// Create a new oneshot channel
    pub fn new() -> Self {
        let (sender, receiver) = bounded(1);
        Self { 
            sender: Some(sender), 
            receiver 
        }
    }

    /// Send a value through the channel
    pub fn send(mut self, value: T) -> Result<(), T> {
        if let Some(sender) = self.sender.take() {
            sender.send(value).map_err(|err| err.into_inner())
        } else {
            Err(value)
        }
    }

    /// Receive the value from the channel
    pub fn recv(self) -> AsyncStream<Result<T, ChannelError>> {
        AsyncStream::with_channel(|stream_sender| {
            std::thread::spawn(move || {
                let result = self.receiver.recv().map_err(|_| ChannelError::Closed);
                let _ = stream_sender.send(result);
            });
        })
    }
}

impl<T> Default for OneshotChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for converting streams into tasks
pub trait IntoTask<T> {
    /// Convert the stream into a task
    fn into_task(self) -> AsyncTask<T>;
}

impl<T> IntoTask<T> for AsyncStream<T>
where
    T: Send + 'static,
{
    fn into_task(self) -> AsyncTask<T> {
        // Create a channel and consume the stream
        let (tx, rx) = crossbeam_channel::bounded(1);
        let mut stream = self;
        std::thread::spawn(move || {
            if let Some(result) = stream.try_next() {
                let _ = tx.send(result);
            }
        });
        AsyncTask::new(rx)
    }
}

// Candle-prefixed type aliases for domain compatibility
pub type CandleChannel<T> = Channel<T>;
pub type CandleOneshotChannel<T> = OneshotChannel<T>;
pub type CandleIntoTask<T> = dyn IntoTask<T>;