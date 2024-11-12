import socket
import struct
from typing import Optional, Tuple
from .constants import RPCCommands
from .exceptions import RPCConnectionError
from .types import TensorParams

class LlamaRPCConnection:
    def __init__(self, host: str = "127.0.0.1", port: int = 50052):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()

    def connect(self) -> None:
        try:
            self.socket.connect((self.host, self.port))
        except socket.error as e:
            raise RPCConnectionError(f"Failed to connect: {e}")

    def _pack_message(self, cmd: RPCCommands, content: bytes) -> bytes:
        return struct.pack('<B', cmd) + struct.pack('<Q', len(content)) + content

    def _receive(self, size: int, timeout: float = 1.0) -> bytes:
        self.socket.settimeout(timeout)
        try:
            data = self.socket.recv(size)
            if not data:
                raise RPCConnectionError("Connection closed by remote host")
            return data
        except socket.timeout:
            raise RPCConnectionError("Receive timeout")

    def alloc_buffer(self, size: int) -> int:
        content = struct.pack('<Q', size)
        packed = self._pack_message(RPCCommands.ALLOC_BUFFER, content)
        self.socket.send(packed)
        
        recv = self._receive(0x20)
        if len(recv) >= 0x10:
            return struct.unpack('<Q', recv[0x8:0x10])[0]
        raise RPCConnectionError("Failed to allocate buffer")

    def get_base(self, ptr: int) -> int:
        content = struct.pack('<Q', ptr)
        packed = self._pack_message(RPCCommands.BUFFER_GET_BASE, content)
        self.socket.send(packed)
        
        recv = self._receive(0x20)
        if len(recv) >= 0x10:
            return struct.unpack('<Q', recv[0x8:0x10])[0]
        raise RPCConnectionError("Failed to get base pointer")

    def copy_tensor(self, tensor_src: bytes, tensor_dst: bytes) -> None:
        payload = tensor_src + tensor_dst
        packed = self._pack_message(RPCCommands.COPY_TENSOR, payload)
        self.socket.send(packed) 