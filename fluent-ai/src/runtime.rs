use std::thread;
use std::sync::mpsc::{channel, Receiver};

pub struct AsyncTask<T> {
    receiver: Receiver<T>,
}

impl<T: Send + 'static> AsyncTask<T> {
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
    {
        let (tx, rx) = channel();
        thread::spawn(move || {
            let res = f();
            let _ = tx.send(res);
        });
        Self { receiver: rx }
    }

    pub fn map<F, U: Send + 'static>(self, f: F) -> AsyncTask<U>
    where
        F: FnOnce(T) -> U + Send + 'static,
    {
        AsyncTask::spawn(move || {
            let value = self.await_blocking();
            f(value)
        })
    }

    pub fn await_blocking(self) -> T {
        self.receiver.recv().unwrap()
    }
}

pub fn run<T>(task: AsyncTask<T>) {
    let _ = task.await_blocking();
}

// AsyncStream is now defined in async_task/stream.rs