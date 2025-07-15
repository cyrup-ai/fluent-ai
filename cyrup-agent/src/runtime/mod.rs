pub mod async_stream;
pub mod async_task;
pub mod channel;
pub mod thread_pool;

pub use async_stream::AsyncStream;
pub use async_task::{spawn_async, AsyncTask};
pub use channel::*;
pub use thread_pool::ThreadPool;

// global single-thread executor reused by AsyncTask
pub(crate) mod executor {
    use super::thread_pool::ThreadPool;
    use crossbeam_channel::{bounded, Receiver};
    use futures_util::task::AtomicWaker;
    use std::sync::Once;

    static INIT: Once = Once::new();
    static mut EXEC: Option<GlobalExecutor> = None;

    pub struct GlobalExecutor {
        pool: ThreadPool,
        waker: AtomicWaker,
        queue: Receiver<(usize, futures_util::future::BoxFuture<'static, ()>)>,
    }

    /// Accessor that bootstraps lazily.
    pub(super) fn global() -> &'static GlobalExecutor {
        unsafe {
            INIT.call_once(|| {
                let pool = ThreadPool::new();
                let (tx, rx) = bounded(1024);
                pool.execute(move || poll_loop(rx.clone()));
                EXEC = Some(GlobalExecutor {
                    pool,
                    waker: AtomicWaker::new(),
                    queue: rx,
                });
            });
            EXEC.as_ref().unwrap()
        }
    }

    /// Register a waker for an AsyncTask receiver.
    pub fn register_waker<R>(rx: R, w: std::task::Waker)
    where
        R: Send + 'static,
    {
        let exec = global();
        exec.waker.register(&w);
        exec.pool.execute(move || {
            // when the value arrives, wake
            let _ = rx;
            exec.waker.wake();
        });
    }

    pub fn enqueue<F>(f: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        global().pool.execute(move || futures_executor::block_on(f));
    }

    fn poll_loop(_rx: Receiver<()>) {}
}
