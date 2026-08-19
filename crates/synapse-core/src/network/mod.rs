mod codec;
mod connection_pool;
mod messages;
mod node;
mod peer_cache;

pub use codec::{ProtocolCodec, SynapseProtocol};
pub use connection_pool::{ConnectionPool, ConnectionState, PoolStats};
pub use messages::{RequestMessage, ResponseMessage};
pub use node::{
    BatchPromptRequest, ChatEvent, NetworkError, NodeStatusSnapshot, PeerSummary, RedialStatus,
    RouteHopView, RoutePlanView, StopReason, SynapseBehaviour, SynapseNode, SynapseRole,
};
pub use peer_cache::PeerCache;

use libp2p::Multiaddr;

pub fn default_listen_addr() -> Multiaddr {
    "/ip4/0.0.0.0/udp/0/quic-v1"
        .parse()
        .expect("static listen address is valid")
}

pub fn parse_listen_addr(input: &str) -> Result<Multiaddr, String> {
    if input.starts_with('/') {
        return input.parse().map_err(|error| format!("{error}"));
    }

    let (host, port) = input
        .rsplit_once(':')
        .ok_or_else(|| "expected either multiaddr or host:port".to_owned())?;
    let port: u16 = port
        .parse()
        .map_err(|_| "port must be a valid u16".to_owned())?;

    if host == "0.0.0.0" {
        return Ok(format!("/ip4/0.0.0.0/udp/{port}/quic-v1")
            .parse()
            .map_err(|error| format!("{error}"))?);
    }

    if host == "127.0.0.1" || host == "localhost" {
        return Ok(format!("/ip4/127.0.0.1/udp/{port}/quic-v1")
            .parse()
            .map_err(|error| format!("{error}"))?);
    }

    Ok(format!("/dns4/{host}/udp/{port}/quic-v1")
        .parse()
        .map_err(|error| format!("{error}"))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_host_port_as_quic_multiaddr() {
        let addr = parse_listen_addr("0.0.0.0:0").expect("address should parse");
        assert_eq!(addr.to_string(), "/ip4/0.0.0.0/udp/0/quic-v1");
    }
}
