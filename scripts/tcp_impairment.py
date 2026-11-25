#!/usr/bin/env python3
"""Bounded loopback TCP latency/bandwidth proxy, never a WAN emulator claim.

Delay applies to each received chunk in each direction. TCP segmentation can
change the number of delays. This does not simulate IP packet loss or netem.
"""
from __future__ import annotations
import random
import socket
import threading
import time

class TcpProxy:
    def __init__(self, target_port: int, delay_ms: float = 0, jitter_ms: float = 0, bytes_per_second: int = 0):
        if not 0 <= delay_ms <= 1000 or not 0 <= jitter_ms <= 1000 or bytes_per_second < 0:
            raise ValueError('invalid bounded impairment settings')
        self.target_port, self.delay_ms, self.jitter_ms, self.rate = target_port, delay_ms, jitter_ms, bytes_per_second
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(('127.0.0.1', 0)); self.listener.listen(16); self.listener.settimeout(.1)
        self.port = self.listener.getsockname()[1]
        self.stopped = threading.Event(); self.lock = threading.Lock(); self.sockets = []
        self.connections = 0; self.delivered_bytes = 0
        self.thread = threading.Thread(target=self.accept, daemon=True); self.thread.start()

    def accept(self):
        while not self.stopped.is_set():
            try: client, _ = self.listener.accept()
            except socket.timeout: continue
            except OSError: break
            with self.lock:
                if self.connections >= 32:
                    client.close(); continue
                self.connections += 1; number = self.connections
            try: remote = socket.create_connection(('127.0.0.1', self.target_port), timeout=2)
            except OSError: client.close(); continue
            client.settimeout(.2); remote.settimeout(.2)
            with self.lock: self.sockets.extend((client, remote))
            for source, destination, direction in [(client, remote, 0), (remote, client, 1)]:
                threading.Thread(target=self.pump, args=(source, destination, number * 2 + direction), daemon=True).start()

    def pump(self, source, destination, seed):
        rng = random.Random(seed)
        try:
            while not self.stopped.is_set():
                try: data = source.recv(4096)
                except socket.timeout: continue
                if not data: break
                delay = max(0, self.delay_ms + rng.uniform(-self.jitter_ms, self.jitter_ms)) / 1000
                if self.rate: delay += len(data) / self.rate
                if self.stopped.wait(delay): break
                destination.sendall(data)
                with self.lock: self.delivered_bytes += len(data)
        except OSError: pass
        finally:
            try: destination.shutdown(socket.SHUT_WR)
            except OSError: pass

    def close(self):
        self.stopped.set(); self.listener.close(); self.thread.join(timeout=1)
        with self.lock:
            for connection in self.sockets:
                try: connection.close()
                except OSError: pass
        return {'kind': 'loopback_tcp_chunk_proxy', 'delay_ms_each_direction_per_chunk': self.delay_ms,
                'jitter_ms_uniform': self.jitter_ms, 'bytes_per_second_per_direction': self.rate or None,
                'connections': self.connections, 'delivered_bytes': self.delivered_bytes,
                'packet_loss_simulated': False}
