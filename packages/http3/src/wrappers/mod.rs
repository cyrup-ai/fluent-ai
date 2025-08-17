//! Wrapper types for external types to implement local traits
//! Solves orphan rule violations for MessageChunk implementations
//!
//! This module is decomposed into logical submodules:
//! - `basic`: Basic type wrappers (Unit, String, Bytes, Generic)
//! - `network`: Network connection wrappers (TcpStream, Upgraded, DNS, etc.)
//! - `http`: HTTP protocol wrappers (Response, Frame, BoxBody, etc.)
//! - `stream`: Stream processing wrappers (HTTP chunks, Download chunks)
//! - `collections`: Collection and utility wrappers (Vec, Option, Tuple, Result)

pub mod basic;
pub mod collections;
pub mod http;
pub mod network;
pub mod stream;

// Re-export all wrapper types for backward compatibility
pub use basic::{BytesWrapper, GenericWrapper, StringWrapper, UnitWrapper};
pub use collections::{OptionWrapper, SocketAddrListWrapper, TupleWrapper, VecWrapper};
pub use http::{BoxBodyWrapper, FrameWrapper, HeaderWrapper, ResponseWrapper};
pub use network::{DnsWrapper, SocketAddrWrapper, TcpStreamWrapper, UpgradedWrapper};
pub use stream::{DownloadChunkWrapper, HttpChunkWrapper, StreamWrapper};
