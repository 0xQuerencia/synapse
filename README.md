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

## running a session

```
synapse host
synapse join
synapse demo
```

Host starts a master and prints a pairing code. Join connects a worker to that master with the code. Demo runs the whole thing on one machine with no network, which is the fastest way to check a build.

Each command takes options. The useful ones: the model option points at any small llama model on Hugging Face, join wants the code the host printed, and the prompt option skips typing. Pass the help flag for the full list.

## the gui

There is also a local web interface in the gui crate. Build it, run the binary it produces, and it prints a local url. You get the console, a live network map, per node hardware stats, and a connect by code panel. A contribute page exists too, for handing friends a pre filled join link.

## config

Relay and bootstrap addresses live in toml files under config. Entries must be full multiaddrs ending in the peer id. Placeholders are skipped at load.

## security

Every peer is untrusted by default. Wire payloads are validated, tensor shapes and dtypes are checked, channels are bounded, and pairing codes never show up in logs after auth. Remote model paths are never accepted for execution.

## status

Working today: sharded inference on LAN and internet, self healing routes when a worker drops, speculative decoding, continuous batching, live telemetry. Next: true batched forward passes and smarter multi worker scheduling.
