//! Local transport evidence only: these tests use three independent libp2p
//! identities on loopback, not physical devices, WAN routers or a public relay.
use futures::StreamExt;
use libp2p::{
    Multiaddr, PeerId, Swarm, SwarmBuilder, identify, ping, relay,
    swarm::{NetworkBehaviour, SwarmEvent, behaviour::toggle::Toggle},
};
use std::time::Duration;
use synapse_core::network::connectivity::{RelayLimits, circuit_address, connection_limits};

#[derive(NetworkBehaviour)]
#[behaviour(prelude = "libp2p::swarm::derive_prelude")]
struct TestBehaviour {
    relay_client: relay::client::Behaviour,
    relay_server: Toggle<relay::Behaviour>,
    identify: identify::Behaviour,
    ping: ping::Behaviour,
    limits: libp2p::connection_limits::Behaviour,
}

fn node(server: bool) -> Swarm<TestBehaviour> {
    node_with_limits(server, RelayLimits::default())
}

fn node_with_limits(server: bool, limits: RelayLimits) -> Swarm<TestBehaviour> {
    SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )
        .unwrap()
        .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)
        .unwrap()
        .with_behaviour(|key, client| {
            let local = key.public().to_peer_id();
            TestBehaviour {
                relay_client: client,
                relay_server: Toggle::from(
                    server
                        .then(|| relay::Behaviour::new(local, limits.to_libp2p_config().unwrap())),
                ),
                identify: identify::Behaviour::new(identify::Config::new(
                    "synapse/transport-test/1".into(),
                    key.public(),
                )),
                ping: ping::Behaviour::new(
                    ping::Config::new().with_interval(Duration::from_millis(100)),
                ),
                limits: connection_limits(),
            }
        })
        .unwrap()
        .with_swarm_config(|config| config.with_idle_connection_timeout(Duration::from_secs(20)))
        .build()
}

#[tokio::test]
async fn relay_reservation_capacity_refuses_extra_peer_without_evicting_existing_one() {
    tokio::time::timeout(Duration::from_secs(20), async {
        let mut relay = node_with_limits(true, RelayLimits {
            max_reservations: 1,
            max_reservations_per_peer: 1,
            ..RelayLimits::default()
        });
        let mut first = node(false);
        let mut extra = node(false);
        let relay_peer = *relay.local_peer_id();
        let extra_peer = *extra.local_peer_id();
        let address = listen(&mut relay).await;
        let mut advertised = address.clone();
        advertised.pop();
        relay.add_external_address(advertised);
        let reservation = address.with(libp2p::multiaddr::Protocol::P2pCircuit);
        first.listen_on(reservation.clone()).unwrap();
        loop {
            tokio::select! {
                _ = relay.select_next_some() => {},
                event = first.select_next_some() => match event {
                    SwarmEvent::NewListenAddr { address, .. } if address.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit)) => break,
                    SwarmEvent::ListenerError { error, .. } => panic!("first reservation failed: {error}"),
                    _ => {},
                }
            }
        }
        extra.listen_on(reservation).unwrap();
        let mut denied = false;
        let mut client_failed = false;
        while !denied || !client_failed {
            tokio::select! {
                event = relay.select_next_some() => {
                    if let SwarmEvent::Behaviour(TestBehaviourEvent::RelayServer(relay::Event::ReservationReqDenied { src_peer_id, status })) = event {
                        assert_eq!(src_peer_id, extra_peer);
                        assert!(matches!(status, relay::StatusCode::ResourceLimitExceeded));
                        denied = true;
                    }
                },
                _ = first.select_next_some() => {},
                event = extra.select_next_some() => match event {
                    SwarmEvent::ListenerError { .. } | SwarmEvent::ListenerClosed { reason: Err(_), .. } => client_failed = true,
                    SwarmEvent::NewListenAddr { address, .. } if address.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit)) => panic!("relay admitted a reservation beyond its configured cap"),
                    _ => {},
                },
            }
        }
        assert!(first.is_connected(&relay_peer));
        assert!(first.listeners().any(|address| address.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit))));
    }).await.expect("relay capacity refusal deadline exceeded");
}

async fn listen(swarm: &mut Swarm<TestBehaviour>) -> Multiaddr {
    swarm
        .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
        .unwrap();
    loop {
        if let SwarmEvent::NewListenAddr { address, .. } = swarm.select_next_some().await {
            return address.with(libp2p::multiaddr::Protocol::P2p(*swarm.local_peer_id()));
        }
    }
}

#[tokio::test]
async fn direct_tcp_noise_authenticates_the_expected_peer() {
    tokio::time::timeout(Duration::from_secs(20), async {
        let mut server = node(false);
        let mut client = node(false);
        let address = listen(&mut server).await;
        let expected = *server.local_peer_id();
        client.dial(address).unwrap();
        let mut authenticated = false;
        loop {
            tokio::select! {
                _ = server.select_next_some() => {},
                event = client.select_next_some() => match event {
                    SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                        assert_eq!(peer_id, expected);
                        assert!(!endpoint.is_relayed());
                        authenticated = true;
                    },
                    SwarmEvent::Behaviour(TestBehaviourEvent::Ping(event)) if event.result.is_ok() => {
                        assert!(authenticated);
                        assert_eq!(event.peer, expected);
                        break;
                    },
                    SwarmEvent::OutgoingConnectionError { error, .. } => panic!("direct connection failed: {error}"),
                    _ => {},
                }
            }
        }
    }).await.expect("direct encrypted loopback connection timed out");
}

#[tokio::test]
async fn relay_circuit_preserves_end_to_end_noise_peer_identity() {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut relay = node(true);
        let mut worker = node(false);
        let mut requester = node(false);
        let relay_address = listen(&mut relay).await;
        let mut advertised = relay_address.clone();
        advertised.pop();
        relay.add_external_address(advertised);
        let target = *worker.local_peer_id();
        worker.listen_on(relay_address.clone().with(libp2p::multiaddr::Protocol::P2pCircuit)).unwrap();
        loop {
            tokio::select! {
                _ = relay.select_next_some() => {},
                event = worker.select_next_some() => match event {
                    SwarmEvent::NewListenAddr { address, .. } if address.iter().any(|part| matches!(part, libp2p::multiaddr::Protocol::P2pCircuit)) => break,
                    SwarmEvent::ListenerError { error, .. } => panic!("relay reservation failed: {error}"),
                    _ => {},
                }
            }
        }
        requester.dial(circuit_address(&relay_address, target).unwrap()).unwrap();
        let mut relayed_connection = false;
        loop {
            tokio::select! {
                _ = relay.select_next_some() => {},
                _ = worker.select_next_some() => {},
                event = requester.select_next_some() => match event {
                    SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } if peer_id == target => {
                        assert!(endpoint.is_relayed());
                        relayed_connection = true;
                    },
                    SwarmEvent::Behaviour(TestBehaviourEvent::Ping(event)) if event.result.is_ok() && event.peer == target => {
                        assert!(relayed_connection);
                        break;
                    },
                    SwarmEvent::OutgoingConnectionError { peer_id: Some(peer), error, .. } if peer == target => panic!("relayed connection failed: {error}"),
                    _ => {},
                }
            }
        }
    }).await.expect("encrypted relayed loopback connection timed out");
}

#[tokio::test]
async fn wrong_pinned_peer_id_fails_noise_authentication() {
    tokio::time::timeout(Duration::from_secs(20), async {
        let mut server = node(false);
        let mut client = node(false);
        let mut address = listen(&mut server).await;
        address.pop();
        let wrong_peer = PeerId::random();
        address.push(libp2p::multiaddr::Protocol::P2p(wrong_peer));
        client.dial(address).unwrap();
        loop {
            tokio::select! {
                _ = server.select_next_some() => {},
                event = client.select_next_some() => match event {
                    SwarmEvent::ConnectionEstablished { .. } => panic!("wrong identity accepted"),
                    SwarmEvent::OutgoingConnectionError { .. } => break,
                    _ => {},
                }
            }
        }
    })
    .await
    .expect("wrong-identity rejection timed out");
}
