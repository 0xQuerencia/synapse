//! Versioned, bounded binary framing. The frame itself is not compressed;
//! tensor payloads carry their separately validated compression metadata.
use super::{RequestMessage, ResponseMessage};
use crate::protocol::MAX_WIRE_MESSAGE_BYTES;
use async_trait::async_trait;
use bincode::Options;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::request_response;
use serde::{Deserialize, Serialize};
use std::{
    io,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

pub const SYNAPSE_PROTOCOL: &str = "/synapse/0.4.0";
pub const MAX_CONTROL_MESSAGE_BYTES: usize = 64 * 1024;
const MAX_PROTOCOL_MESSAGE_BYTES: usize = MAX_WIRE_MESSAGE_BYTES;
const MAX_DECODING_BYTES: usize = 2 * MAX_WIRE_MESSAGE_BYTES;
const FRAME_TIMEOUT: Duration = Duration::from_secs(15);
const FRAME_MARKER: u8 = 0xff;
static DECODING_BYTES: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Default)]
pub struct ProtocolCodec;
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynapseProtocol;
impl AsRef<str> for SynapseProtocol {
    fn as_ref(&self) -> &str {
        SYNAPSE_PROTOCOL
    }
}

#[derive(Clone, Copy)]
enum FrameKind {
    Request,
    Response,
}
impl FrameKind {
    fn maximum(self, ordinal: u32) -> usize {
        // These stable ordinals are pinned by golden vectors. New control
        // variants append to enums and retain the smaller control limit.
        let data = match self {
            Self::Request => ordinal == 1,
            Self::Response => ordinal == 1 || ordinal == 5,
        };
        if data {
            MAX_PROTOCOL_MESSAGE_BYTES
        } else {
            MAX_CONTROL_MESSAGE_BYTES
        }
    }
}

struct DecodeBudget<'a> {
    bytes: usize,
    used: &'a AtomicUsize,
}
impl DecodeBudget<'static> {
    fn reserve(bytes: usize) -> io::Result<Self> {
        Self::reserve_for(&DECODING_BYTES, bytes)
    }
}
impl<'a> DecodeBudget<'a> {
    fn reserve_for(used: &'a AtomicUsize, bytes: usize) -> io::Result<Self> {
        used.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current
                .checked_add(bytes)
                .filter(|next| *next <= MAX_DECODING_BYTES)
        })
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                "aggregate wire decoding budget exhausted",
            )
        })?;
        Ok(Self { bytes, used })
    }
}
impl Drop for DecodeBudget<'_> {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[async_trait]
impl request_response::Codec for ProtocolCodec {
    type Protocol = SynapseProtocol;
    type Request = RequestMessage;
    type Response = ResponseMessage;

    async fn read_request<T>(&mut self, _: &Self::Protocol, io: &mut T) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let message: Self::Request = read_frame(io, FrameKind::Request).await?;
        message.validate_wire().map_err(invalid)?;
        Ok(message)
    }
    async fn read_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let message: Self::Response = read_frame(io, FrameKind::Response).await?;
        message.validate_wire().map_err(invalid)?;
        Ok(message)
    }
    async fn write_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        message: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        message.validate_wire().map_err(invalid)?;
        write_frame(io, &message, FrameKind::Request).await
    }
    async fn write_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        message: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        message.validate_wire().map_err(invalid)?;
        write_frame(io, &message, FrameKind::Response).await
    }
}

fn invalid(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

fn options(limit: usize) -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(limit as u64)
        .reject_trailing_bytes()
}

async fn read_frame<T, M>(io: &mut T, kind: FrameKind) -> io::Result<M>
where
    T: AsyncRead + Unpin + Send,
    M: for<'de> Deserialize<'de>,
{
    tokio::time::timeout(FRAME_TIMEOUT, async {
        let mut header = [0_u8; 5];
        io.read_exact(&mut header).await?;
        if header[0] != FRAME_MARKER {
            return Err(invalid("unsupported frame marker"));
        }
        let len = u32::from_le_bytes(header[1..].try_into().expect("four length bytes")) as usize;
        if !(4..=MAX_PROTOCOL_MESSAGE_BYTES).contains(&len) {
            return Err(invalid("frame length outside limits"));
        }
        // Read the enum tag before allocating an attacker-controlled large body.
        let mut tag = [0_u8; 4];
        io.read_exact(&mut tag).await?;
        if len > kind.maximum(u32::from_le_bytes(tag)) {
            return Err(invalid("control frame exceeds 64 KiB"));
        }
        // Reserve twice the frame size because deserialization may briefly own
        // both the encoded frame and decoded fields.
        let _budget = DecodeBudget::reserve(len.saturating_mul(2))?;
        let mut bytes = vec![0_u8; len];
        bytes[..4].copy_from_slice(&tag);
        io.read_exact(&mut bytes[4..]).await?;
        options(len).deserialize(&bytes).map_err(invalid)
    })
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "wire frame deadline exceeded"))?
}

async fn write_frame<T, M>(io: &mut T, message: &M, kind: FrameKind) -> io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
    M: Serialize,
{
    let size = options(MAX_PROTOCOL_MESSAGE_BYTES)
        .serialized_size(message)
        .map_err(invalid)?;
    if size > MAX_PROTOCOL_MESSAGE_BYTES as u64 {
        return Err(invalid("wire message exceeds limit"));
    }
    let bytes = options(MAX_PROTOCOL_MESSAGE_BYTES)
        .serialize(message)
        .map_err(invalid)?;
    let tag: [u8; 4] = bytes
        .get(..4)
        .ok_or_else(|| invalid("missing variant tag"))?
        .try_into()
        .expect("four tag bytes");
    if bytes.len() > kind.maximum(u32::from_le_bytes(tag)) {
        return Err(invalid("control frame exceeds 64 KiB"));
    }
    tokio::time::timeout(FRAME_TIMEOUT, async {
        io.write_all(&[FRAME_MARKER]).await?;
        io.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
        io.write_all(&bytes).await?;
        io.close().await
    })
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "wire write deadline exceeded"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::io::Cursor;
    use libp2p::request_response::Codec;

    fn frame(body: &[u8]) -> Vec<u8> {
        let mut bytes = vec![FRAME_MARKER];
        bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
        bytes.extend_from_slice(body);
        bytes
    }
    #[tokio::test]
    async fn health_ping_matches_golden_wire_bytes() {
        let mut output = Cursor::new(Vec::new());
        ProtocolCodec
            .write_request(&SynapseProtocol, &mut output, RequestMessage::HealthPing)
            .await
            .unwrap();
        assert_eq!(output.into_inner(), vec![0xff, 4, 0, 0, 0, 7, 0, 0, 0]);
        let mut input = Cursor::new(vec![0xff, 4, 0, 0, 0, 7, 0, 0, 0]);
        assert!(matches!(
            ProtocolCodec
                .read_request(&SynapseProtocol, &mut input)
                .await
                .unwrap(),
            RequestMessage::HealthPing
        ));
    }
    #[tokio::test]
    async fn malformed_frames_fail_without_large_allocations() {
        let mut cases = vec![
            vec![],
            vec![0xff],
            vec![0, 4, 0, 0, 0, 7, 0, 0, 0],
            vec![0xff, 0xff, 0xff, 0xff, 0xff],
            frame(&[7, 0, 0, 0, 9]),
            frame(&[0xff, 0xff, 0xff, 0xff]),
        ];
        // ClearCache contains a String. A malicious u64 length must not
        // become a multi-gigabyte allocation from a tiny received frame.
        let mut hostile = 2_u32.to_le_bytes().to_vec();
        hostile.extend_from_slice(&u64::MAX.to_le_bytes());
        cases.push(frame(&hostile));
        for case in cases {
            assert!(
                ProtocolCodec
                    .read_request(&SynapseProtocol, &mut Cursor::new(case))
                    .await
                    .is_err()
            );
        }
    }
    #[tokio::test]
    async fn oversized_control_is_rejected_after_tag() {
        let mut header = vec![FRAME_MARKER];
        header.extend_from_slice(&((MAX_CONTROL_MESSAGE_BYTES + 1) as u32).to_le_bytes());
        header.extend_from_slice(&7_u32.to_le_bytes());
        let error = ProtocolCodec
            .read_request(&SynapseProtocol, &mut Cursor::new(header))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
    #[tokio::test]
    async fn decoded_fields_are_validated_before_dispatch() {
        let invalid_message = RequestMessage::ClearCache(crate::ClearCache {
            trace_id: "../escape".into(),
        });
        let body = bincode::serialize(&invalid_message).unwrap();
        assert!(
            ProtocolCodec
                .read_request(&SynapseProtocol, &mut Cursor::new(frame(&body)))
                .await
                .is_err()
        );
    }
    #[test]
    fn protocol_and_codec_limits_match() {
        assert_eq!(
            MAX_PROTOCOL_MESSAGE_BYTES,
            crate::protocol::MAX_WIRE_MESSAGE_BYTES
        );
    }
    #[test]
    fn decoding_budget_is_released_even_on_failure() {
        let used = AtomicUsize::new(0);
        assert!(DecodeBudget::reserve_for(&used, MAX_DECODING_BYTES + 1).is_err());
        {
            let _guard = DecodeBudget::reserve_for(&used, MAX_DECODING_BYTES).unwrap();
            assert!(DecodeBudget::reserve_for(&used, 1).is_err());
        }
        assert_eq!(used.load(Ordering::Acquire), 0);
        assert!(DecodeBudget::reserve_for(&used, MAX_DECODING_BYTES).is_ok());
    }
}
