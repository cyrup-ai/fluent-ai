use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use crossbeam_queue::ArrayQueue;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

type Job = Box<dyn FnOnce() + Send + 'static>;

const MAX_IDLE: usize = 64;
static IDLE_RING: ArrayQueue<thread::Thread> = ArrayQueue::new(MAX_IDLE);

#[derive(Clone)]
pub struct ThreadPool {
    injector: Arc<Injector<Job>>,
    stealers: Arc<Vec<Stealer<Job>>>,
    rr: Arc<AtomicUsize>,
}

impl ThreadPool {
    pub fn new() -> Self {
        let cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let injector = Arc::new(Injector::new());
        let mut stealers = Vec::with_capacity(cpus);

        for id in 0..cpus {
            let local = Worker::new_fifo();
            stealers.push(local.stealer());

            let inj = Arc::clone(&injector);
            let stealers = stealers.clone();

            if let Err(e) = thread::Builder::new()
                .name(format!("rig-worker-{id}"))
                .spawn(move || worker_loop(local, inj, stealers)) {
                eprintln!("Failed to spawn worker thread {}: {}", id, e);
                // Continue with fewer workers rather than panicking
            }
        }

        Self {
            injector,
            stealers: Arc::new(stealers),
            rr: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[inline(always)]
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.injector.push(Box::new(f));

        // try wake an idle thread
        if let Some(t) = IDLE_RING.pop() {
            t.unpark();
            return;
        }

        // otherwise prod next worker (round-robin)
        let i = self.rr.fetch_add(1, Ordering::Relaxed) % self.stealers.len();
        if let Some(t) = self.stealers[i]
            .steal()
            .success()
            .and_then(|_| Some(thread::current()))
        {
            t.unpark();
        }
    }
}

fn worker_loop(local: Worker<Job>, inj: Arc<Injector<Job>>, stealers: Vec<Stealer<Job>>) {
    loop {
        // 1) local FIFO
        if let Some(job) = local.pop() {
            (job)();
            continue;
        }
        // 2) global injector
        if let Steal::Success(job) = inj.steal_batch_and_pop(&local) {
            (job)();
            continue;
        }
        // 3) steal from peers
        for stealer in &stealers {
            if let Steal::Success(job) = stealer.steal_batch(&local) {
                (job)();
                continue;
            }
        }
        // 4) idle
        if IDLE_RING.push(thread::current()).is_ok() {
            thread::park();
        } else {
            thread::sleep(Duration::from_micros(50));
        }
    }
}
