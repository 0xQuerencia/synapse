mod codec;
mod connection_pool;
pub mod connectivity;
mod messages;
mod node;
mod peer_cache;
pub mod quota;
pub mod session;
pub mod stream;
pub mod worker;

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

    if let Ok(socket) = input.parse::<std::net::SocketAddr>() {
        let host = match socket.ip() {
            std::net::IpAddr::V4(ip) => format!("/ip4/{ip}"),
            std::net::IpAddr::V6(ip) => format!("/ip6/{ip}"),
        };
        return format!("{host}/udp/{}/quic-v1", socket.port())
            .parse()
            .map_err(|error| format!("{error}"));
    }
    let (host, port) = input
        .rsplit_once(':')
        .ok_or_else(|| "expected multiaddr, IP:port, [IPv6]:port or hostname:port".to_owned())?;
    let port: u16 = port
        .parse()
        .map_err(|_| "port must be a valid u16".to_owned())?;
    if host == "localhost" {
        return format!("/ip4/127.0.0.1/udp/{port}/quic-v1")
            .parse()
            .map_err(|error| format!("{error}"));
    }
    if host.is_empty()
        || host.len() > 253
        || !host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".-".contains(&byte))
    {
        return Err("invalid hostname; IPv6 addresses require brackets".into());
    }
    format!("/dns4/{host}/udp/{port}/quic-v1")
        .parse()
        .map_err(|error| format!("{error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_host_port_as_quic_multiaddr() {
        let addr = parse_listen_addr("0.0.0.0:0").expect("address should parse");
        assert_eq!(addr.to_string(), "/ip4/0.0.0.0/udp/0/quic-v1");
    }

    #[test]
    fn explicit_ip_addresses_do_not_become_dns_queries() {
        assert_eq!(
            parse_listen_addr("192.0.2.1:4001").unwrap().to_string(),
            "/ip4/192.0.2.1/udp/4001/quic-v1"
        );
        assert_eq!(
            parse_listen_addr("[::1]:4001").unwrap().to_string(),
            "/ip6/::1/udp/4001/quic-v1"
        );
        assert!(parse_listen_addr("::1:4001").is_err());
        assert!(parse_listen_addr("user:secret@example.com:4001").is_err());
    }
}
