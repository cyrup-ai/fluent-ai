use std::str;

/// Streaming line-by-line decoder (handles `\n`, `\r`, `\r\n` & mixed streams)
/// with **zero heap re-allocation on the hot path**.
pub struct LineDecoder {
    buf: Vec<u8>,
    pending_cr: Option<usize>, // index of a lone '\r' we’re waiting to resolve
}

impl Default for LineDecoder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl LineDecoder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            pending_cr: None,
        }
    }

    /// Feed a chunk of bytes – returns every **complete** line parsed so far.
    ///
    /// Empty lines are preserved.  Incomplete tail is kept in the internal
    /// buffer until the next `decode`/`flush`.
    pub fn decode(&mut self, chunk: &[u8]) -> Vec<String> {
        if !chunk.is_empty() {
            self.buf.extend_from_slice(chunk);
        }

        let mut out = Vec::new();
        let mut cursor = 0;

        while let Some(nl) = find_newline(&self.buf[cursor..], self.pending_cr.map(|i| i - cursor))
        {
            // Handle stray '\r' bookkeeping
            if nl.carriage && self.pending_cr.is_none() {
                self.pending_cr = Some(cursor + nl.idx);
                cursor += nl.idx + 1;
                continue;
            }
            if let Some(cr) = self.pending_cr {
                if nl.idx + cursor != cr + 1 || nl.carriage {
                    push_line(&self.buf[..cr.saturating_sub(1)], &mut out);
                    shift_left(&mut self.buf, cr);
                    cursor = 0;
                    self.pending_cr = None;
                    continue;
                }
            }

            let end = if self.pending_cr.is_some() {
                nl.preceding - 1
            } else {
                nl.preceding
            };
            push_line(&self.buf[..end], &mut out);
            shift_left(&mut self.buf, cursor + nl.idx + 1);
            cursor = 0;
            self.pending_cr = None;
        }

        out
    }

    /// Flush whatever is left (treat buffer end as newline).
    #[inline(always)]
    pub fn flush(&mut self) -> Vec<String> {
        self.decode(b"\n")
    }
}

/* ------------------------------------------------------------------ helpers */

#[inline(always)]
fn push_line(slice: &[u8], out: &mut Vec<String>) {
    if slice.is_empty() {
        out.push(String::new());
    } else {
        out.push(decode_utf8(slice));
    }
}

/// Move `buf[start..]` to the front **in-place** and truncate.
#[inline(always)]
fn shift_left(buf: &mut Vec<u8>, start: usize) {
    let len = buf.len();
    if start < len {
        buf.copy_within(start.., 0);
    }
    buf.truncate(len - start);
}

struct Nl {
    preceding: usize,
    idx: usize,
    carriage: bool,
}

/// Return position of next `\n` or `\r` starting at `search_offset`.
fn find_newline(bytes: &[u8], search_offset: Option<usize>) -> Option<Nl> {
    const NL: u8 = 0x0a;
    const CR: u8 = 0x0d;
    let start = search_offset.unwrap_or(0);
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if b == NL {
            return Some(Nl {
                preceding: i,
                idx: i,
                carriage: false,
            });
        }
        if b == CR {
            return Some(Nl {
                preceding: i,
                idx: i,
                carriage: true,
            });
        }
    }
    None
}

/// Find the index of a double newline sequence (\n\n or \r\n\r\n) in SSE streams
pub fn find_double_newline_index(bytes: &[u8]) -> Option<usize> {
    const NL: u8 = 0x0a; // '\n'
    const CR: u8 = 0x0d; // '\r'

    for i in 0..bytes.len().saturating_sub(1) {
        match (bytes[i], bytes.get(i + 1)) {
            // \n\n
            (NL, Some(&NL)) => return Some(i),
            // \r\n\r\n (check if we have enough bytes)
            (CR, Some(&NL)) if i + 3 < bytes.len() => {
                if bytes[i + 2] == CR && bytes[i + 3] == NL {
                    return Some(i);
                }
            }
            _ => continue,
        }
    }
    None
}

#[inline(always)]
fn decode_utf8(bytes: &[u8]) -> String {
    match str::from_utf8(bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => String::from_utf8_lossy(bytes).into_owned(),
    }
}

/* ------------------------------------------------------------------ tests */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::line::find_double_newline_index as ddn;

    fn lines(chunks: &[&str], flush: bool) -> Vec<String> {
        let mut ld = LineDecoder::new();
        let mut v = Vec::new();
        for c in chunks {
            v.extend(ld.decode(c.as_bytes()));
        }
        if flush {
            v.extend(ld.flush());
        }
        v
    }

    #[test]
    fn basic() {
        assert_eq!(lines(&["foo", " bar\nbaz"], false), ["foo bar"]);
    }
    #[test]
    fn basic_cr() {
        assert_eq!(lines(&["foo", " bar\r\nbaz"], false), ["foo bar"]);
        assert_eq!(lines(&["foo", " bar\r\nbaz"], true), ["foo bar", "baz"]);
    }
    #[test]
    fn trailing() {
        assert_eq!(
            lines(&["foo", " bar", "baz\n", "thing\n"], false),
            ["foo barbaz", "thing"]
        );
    }
    #[test]
    fn escaped() {
        assert_eq!(lines(&["foo", " bar\\nbaz\n"], false), ["foo bar\\nbaz"]);
    }
    #[test]
    fn flush_empty() {
        assert!(lines(&[], true).is_empty());
    }
    #[test]
    fn dbl_nl() {
        assert_eq!(ddn("foo\n\nbar".as_bytes()), 5);
    }
}
