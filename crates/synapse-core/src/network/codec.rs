use std::io;

use async_trait::async_trait;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::request_response;
use serde::{Deserialize, Serialize};

use super::{RequestMessage, ResponseMessage};
use crate::protocol::MAX_WIRE_MESSAGE_BYTES;

pub const SYNAPSE_PROTOCOL: &str = "/synapse/0.2.0";
const MAX_PROTOCOL_MESSAGE_BYTES: usize = MAX_WIRE_MESSAGE_BYTES;

#[derive(Clone, Default)]
pub struct ProtocolCodec;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynapseProtocol;

impl AsRef<str> for SynapseProtocol {
    fn as_ref(&self) -> &str {
        SYNAPSE_PROTOCOL
    }
}

#[async_trait]
impl request_response::Codec for ProtocolCodec {
    type Protocol = SynapseProtocol;
    type Request = RequestMessage;
    type Response = ResponseMessage;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_bincode(io).await
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_bincode(io).await
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_bincode(io, &req).await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_bincode(io, &res).await
    }
}

async fn read_bincode<T, M>(io: &mut T) -> io::Result<M>
where
    T: AsyncRead + Unpin + Send,
    M: for<'de> Deserialize<'de>,
{
    let mut bytes = Vec::new();
    io.take(MAX_PROTOCOL_MESSAGE_BYTES as u64)
        .read_to_end(&mut bytes)
        .await?;

    bincode::deserialize(&bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

async fn write_bincode<T, M>(io: &mut T, message: &M) -> io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
    M: Serialize,
{
    let bytes = bincode::serialize(message)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

    if bytes.len() > MAX_PROTOCOL_MESSAGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "protocol message exceeds size limit",
        ));
    }

    io.write_all(&bytes).await?;
    io.close().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_and_codec_limits_match() {
        assert_eq!(
            MAX_PROTOCOL_MESSAGE_BYTES,
            crate::protocol::MAX_WIRE_MESSAGE_BYTES
        );
    }
}
