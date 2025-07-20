#![allow(dead_code)]

use std::time::Duration;

use crossbeam_channel::{
    Receiver, RecvError, Sender, TryIter, TryRecvError, TrySendError, bounded as cb_bounded,
    unbounded as cb_unbounded,
};

#[derive(Clone)]
pub struct Tx<T>(Sender<T>);

impl<T: Send + 'static> Tx<T> {
    #[inline(always)]
    pub fn try_send(&self, msg: T) -> Result<(), TrySendError<T>> {
        self.0.try_send(msg)
    }

    #[inline(always)]
    pub fn send(&self, msg: T) -> Result<(), TrySendError<T>> {
        self.try_send(msg)
    }

    #[inline(always)]
    pub fn send_timeout(&self, msg: T, timeout: Duration) -> Result<(), TrySendError<T>> {
        self.0.send_timeout(msg, timeout)
    }
}

pub struct Rx<T>(Receiver<T>);

impl<T> Rx<T> {
    #[inline(always)]
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        self.0.try_recv()
    }

    #[inline(always)]
    pub fn try_iter(&self) -> TryIter<'_, T> {
        self.0.try_iter()
    }

    #[inline(always)]
    pub fn recv(&self) -> Result<T, RecvError> {
        self.0.recv()
    }

    #[inline(always)]
    pub fn recv_timeout(&self, duration: Duration) -> Result<T, RecvError> {
        self.0.recv_timeout(duration)
    }
}

#[inline(always)]
pub fn unbounded<T>() -> (Tx<T>, Rx<T>) {
    let (s, r) = cb_unbounded();
    (Tx(s), Rx(r))
}

#[inline(always)]
pub fn bounded<T>(cap: usize) -> (Tx<T>, Rx<T>) {
    let (s, r) = cb_bounded(cap);
    (Tx(s), Rx(r))
}
