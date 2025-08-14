//! Concurrency primitives and utilities

use std::sync::Arc;

use fluent_ai_async::AsyncTask;
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::core::ChannelError;

/// A multi-producer, single-consumer channel for sending values between tasks
pub struct Channel<T> {
    sender: mpsc::Sender<T>,
    receiver: Arc<Mutex<mpsc::Receiver<T>>>,
}

impl<T> Clone for Channel<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
        }
    }
}

impl<T: Send + 'static> Channel<T> {
    /// Create a new channel with the given buffer size
    pub fn new(buffer: usize) -> Self {
        let (sender, receiver) = mpsc::channel(buffer);
        Self {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    /// Send a value into the channel
    pub async fn send(&self, value: T) -> Result<(), ChannelError> {
        self.sender
            .send(value)
            .await
            .map_err(|_| ChannelError::SendError)
    }

    /// Receive the next value from the channel
    pub async fn recv(&self) -> Result<T, ChannelError> {
        self.receiver
            .lock()
            .await
            .recv()
            .await
            .ok_or(ChannelError::Closed)
    }

    /// Create a new receiver that can be used to receive values from this channel
    pub fn subscribe(
        &self,
    ) -> impl futures_util::Stream<Item = Result<T, ChannelError>> + Send + 'static {
        let receiver = self.receiver.clone();
        async_stream::stream! {
            let mut receiver = receiver.lock().await;
            while let Some(value) = receiver.recv().await {
                yield Ok(value);
            }
            yield Err(ChannelError::Closed);
        }
    }
}

/// A oneshot channel for sending a single value between tasks
pub struct OneshotChannel<T> {
    sender: oneshot::Sender<T>,
    receiver: oneshot::Receiver<T>,
}

impl<T> OneshotChannel<T> {
    /// Create a new oneshot channel
    pub fn new() -> Self {
        let (sender, receiver) = oneshot::channel();
        Self { sender, receiver }
    }

    /// Send a value through the channel
    pub fn send(self, value: T) -> Result<(), T> {
        self.sender.send(value)
    }

    /// Receive the value from the channel
    pub async fn recv(self) -> Result<T, ChannelError> {
        self.receiver.await.map_err(|_| ChannelError::Closed)
    }
}

impl<T> Default for OneshotChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for converting futures into tasks
pub trait IntoTask<T> {
    /// Convert the future into a task
    fn into_task(self) -> AsyncTask<T>;
}

impl<F, T> IntoTask<T> for F
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    fn into_task(self) -> AsyncTask<T> {
        // Create a channel and spawn the future
        let (tx, rx) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Handle::current();
            let result = runtime.block_on(self);
            let _ = tx.send(result);
        });
        AsyncTask::new(rx)
    }
}
