"""
distributed_bus.py — Network-capable message bus for distributed brain regions.

Extends the local MessageBus to support brain regions running on different
machines. Uses a simple TCP socket protocol for cross-machine communication.

Architecture:
    Local Machine (fast path):     GPU Server (slow path):
      SensoryCortex                  Striatum (training)
      MotorCortex                    PrefrontalCortex (LLM)
      AmygdalaThalamus               DreamLoop (overnight)

    Connected via DistributedBus over TCP.

Usage:
    # On local machine (coordinator):
    from brain.scaling.distributed_bus import DistributedBus
    bus = DistributedBus(role="coordinator", port=9500)
    bus.start()

    # On GPU server (worker):
    bus = DistributedBus(role="worker", coordinator_host="192.168.1.10", port=9500)
    bus.start()
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from brain.message_bus import MessageBus, BrainMessage


class DistributedBus(MessageBus):
    """
    Network-capable extension of the local MessageBus.

    In coordinator mode: listens for remote workers and forwards messages.
    In worker mode: connects to coordinator and proxies local messages.
    """

    def __init__(
        self,
        role: str = "local",        # "local", "coordinator", or "worker"
        coordinator_host: str = "127.0.0.1",
        port: int = 9500,
        history_size: int = 1000,
    ):
        super().__init__(history_size=history_size)
        self.role = role
        self.coordinator_host = coordinator_host
        self.port = port

        self._remote_regions: Set[str] = set()
        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._connections: List[socket.socket] = []
        self._running = False
        self._net_thread: Optional[threading.Thread] = None
        self._outbound: queue.Queue = queue.Queue(maxsize=1000)

    def start(self) -> None:
        """Start the network layer."""
        if self.role == "local":
            return  # Local mode, no networking

        self._running = True

        if self.role == "coordinator":
            self._start_coordinator()
        elif self.role == "worker":
            self._start_worker()

    def stop(self) -> None:
        """Stop the network layer."""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        if self._client_socket:
            try:
                self._client_socket.close()
            except Exception:
                pass
        for conn in self._connections:
            try:
                conn.close()
            except Exception:
                pass

    def register_remote(self, region_name: str) -> None:
        """Register a brain region as running on a remote machine."""
        self._remote_regions.add(region_name)

    def send(self, msg: BrainMessage) -> None:
        """Send a message, routing to remote if recipient is on another machine."""
        # Always deliver locally first
        super().send(msg)

        # Forward to remote if needed
        if msg.target in self._remote_regions or msg.target == "broadcast":
            self._enqueue_remote(msg)

    def _enqueue_remote(self, msg: BrainMessage) -> None:
        """Queue a message for network transmission."""
        try:
            payload = {
                "source": msg.source,
                "target": msg.target,
                "priority": int(msg.priority),
                "msg_type": msg.msg_type,
                "payload": _safe_serialize(msg.payload),
                "timestamp": msg.timestamp,
            }
            self._outbound.put_nowait(json.dumps(payload))
        except queue.Full:
            pass  # Drop if queue is full

    # ── Coordinator ───────────────────────────────────────────────────

    def _start_coordinator(self) -> None:
        """Start listening for worker connections."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(("0.0.0.0", self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)

        self._net_thread = threading.Thread(
            target=self._coordinator_loop, daemon=True
        )
        self._net_thread.start()

    def _coordinator_loop(self) -> None:
        """Accept connections and relay messages."""
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                self._connections.append(conn)
                # Start a reader thread for each connection
                threading.Thread(
                    target=self._read_from_connection,
                    args=(conn,),
                    daemon=True,
                ).start()
            except socket.timeout:
                pass
            except Exception:
                break

            # Send queued outbound messages to all connections
            self._flush_outbound()

    # ── Worker ────────────────────────────────────────────────────────

    def _start_worker(self) -> None:
        """Connect to coordinator."""
        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._client_socket.connect((self.coordinator_host, self.port))
        except ConnectionRefusedError:
            self._client_socket = None
            return

        self._net_thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self._net_thread.start()

    def _worker_loop(self) -> None:
        """Worker: read from coordinator and send outbound."""
        while self._running and self._client_socket:
            try:
                # Read incoming messages
                data = self._recv_message(self._client_socket)
                if data:
                    msg_dict = json.loads(data)
                    msg = BrainMessage(
                        source=msg_dict["source"],
                        target=msg_dict["target"],
                        priority=msg_dict.get("priority", 0),
                        msg_type=msg_dict.get("msg_type", "remote"),
                        payload=msg_dict.get("payload", {}),
                    )
                    # Deliver locally
                    super().send(msg)
            except Exception:
                break

            self._flush_outbound()

    # ── Network I/O ───────────────────────────────────────────────────

    def _flush_outbound(self) -> None:
        """Send all queued messages to connected sockets."""
        while not self._outbound.empty():
            try:
                payload = self._outbound.get_nowait()
                data = payload.encode("utf-8")
                header = struct.pack("!I", len(data))

                targets = self._connections if self.role == "coordinator" else (
                    [self._client_socket] if self._client_socket else []
                )
                for sock in targets:
                    try:
                        sock.sendall(header + data)
                    except Exception:
                        pass
            except queue.Empty:
                break

    def _read_from_connection(self, conn: socket.socket) -> None:
        """Read messages from a connection and deliver locally."""
        while self._running:
            try:
                data = self._recv_message(conn)
                if not data:
                    break
                msg_dict = json.loads(data)
                msg = BrainMessage(
                    source=msg_dict["source"],
                    target=msg_dict["target"],
                    priority=msg_dict.get("priority", 0),
                    data=msg_dict.get("data", {}),
                )
                super().send(msg)
            except Exception:
                break

    def _recv_message(self, sock: socket.socket) -> Optional[str]:
        """Receive a length-prefixed message from a socket."""
        sock.settimeout(0.5)
        try:
            header = self._recv_exact(sock, 4)
            if not header:
                return None
            length = struct.unpack("!I", header)[0]
            if length > 10 * 1024 * 1024:  # 10MB limit
                return None
            data = self._recv_exact(sock, length)
            return data.decode("utf-8") if data else None
        except socket.timeout:
            return None

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        chunks = []
        received = 0
        while received < n:
            chunk = sock.recv(min(n - received, 4096))
            if not chunk:
                return None
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    # ── Reporting ─────────────────────────────────────────────────────

    def network_stats(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "running": self._running,
            "connections": len(self._connections),
            "remote_regions": list(self._remote_regions),
            "outbound_queued": self._outbound.qsize(),
        }


def _safe_serialize(data: Any) -> Any:
    """Convert data to JSON-safe types."""
    if data is None:
        return None
    if isinstance(data, (int, float, str, bool)):
        return data
    if isinstance(data, dict):
        return {str(k): _safe_serialize(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_safe_serialize(v) for v in data]
    if hasattr(data, "tolist"):
        return data.tolist()
    return str(data)
