# synapse

Run one language model across many computers, as if they were one computer.

Synapse cuts a model into layers and hands the pieces out over a peer to peer network. Your laptop holds the bottom of the network, a desktop across the room holds the top, and tokens flow between them while you chat. No cluster to rent, no cloud bill, no weights leaving your machines.

## how it works

1. The master owns the embedding, the first chunk of layers, and the output head. It runs the show: pairing codes, routing, sampling.
2. Workers take the rest of the layers. A token hops from the master through each worker and back, one hidden state at a time.
3. An optional relay on a public machine lets people behind home routers find each other. It relays encrypted traffic and seeds discovery.

Peers find each other three ways. On one network, multicast DNS just works. Across the internet, a DHT keyed by model id locates masters serving the model you want. When both sides sit behind NAT, relay circuits plus hole punching connect them.

## getting started

```
git clone <repo url>
cd synapse
cargo build
```

Rust is the only requirement. Candle does the math, libp2p does the networking, and everything is tuned for the cpu you build on, so a plain build is already quick.
