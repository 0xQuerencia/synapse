# synapse

Run one language model across many computers, as if they were one computer.

Synapse cuts a model into layers and hands the pieces out over a peer to peer network. Your laptop holds the bottom of the network, a desktop across the room holds the top, and tokens flow between them while you chat. No cluster to rent, no cloud bill, no weights leaving your machines.

## how it works

1. The master owns the embedding, the first chunk of layers, and the output head. It runs the show: pairing codes, routing, sampling.
2. Workers take the rest of the layers. A token hops from the master through each worker and back, one hidden state at a time.
3. An optional relay on a public machine lets people behind home routers find each other. It relays encrypted traffic and seeds discovery.
