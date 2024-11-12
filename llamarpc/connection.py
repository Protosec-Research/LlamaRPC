import struct
import socket
from typing import Optional, Tuple, Any
from .constants import RPCCommands
from .exceptions import RPCConnectionError
from .types import (
    AllocBufferRequest, AllocBufferResponse,
    GetAlignmentResponse, GetMaxSizeResponse,
    BufferGetBaseRequest, BufferGetBaseResponse,
    FreeBufferRequest, BufferClearRequest,
    GetTensorRequest, CopyTensorRequest,
    CopyTensorResponse, GraphComputeResponse,
    GetDeviceMemoryResponse, RPCTensor
)
from .tensor import TensorBuilder

class LlamaRPCConnection:
    def __init__(self, host: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        
    def _pack_message(self, cmd: RPCCommands, payload: bytes) -> bytes:
        msg_len = len(payload)
        header = struct.pack('<II', cmd, msg_len)
        print(f"Sending command: {cmd.name} with payload length: {msg_len}")
        return header + payload

    def _receive(self, expected_size: int) -> bytes:
        data = bytearray()
        while len(data) < expected_size:
            chunk = self.socket.recv(expected_size - len(data))
            if not chunk:
                raise RPCConnectionError("Connection closed by remote host")
            data.extend(chunk)
        return bytes(data)

    def _receive_response(self, expected_size: int) -> Tuple[int, bytes]:
        header = self._receive(8)
        status, size = struct.unpack('<II', header)
        if size > 0:
            payload = self._receive(size)
        else:
            payload = b''
        return status, payload

    def alloc_buffer(self, size: int) -> AllocBufferResponse:
        req = AllocBufferRequest(size=size)
        payload = struct.pack('<Q', req.size)
        packed = self._pack_message(RPCCommands.ALLOC_BUFFER, payload)
        self.socket.send(packed)
        
        status, response = self._receive_response(16)
        if status == 0:
            remote_ptr, remote_size = struct.unpack('<QQ', response)
            return AllocBufferResponse(remote_ptr=remote_ptr, remote_size=remote_size)
        raise RPCConnectionError("Failed to allocate buffer")

    def get_alignment(self) -> GetAlignmentResponse:
        packed = self._pack_message(RPCCommands.GET_ALIGNMENT, b'')
        self.socket.send(packed)
        
        status, response = self._receive_response(8)
        if status == 0:
            alignment = struct.unpack('<Q', response)[0]
            return GetAlignmentResponse(alignment=alignment)
        raise RPCConnectionError("Failed to get alignment")

    def get_max_size(self) -> GetMaxSizeResponse:
        packed = self._pack_message(RPCCommands.GET_MAX_SIZE, b'')
        self.socket.send(packed)
        
        status, response = self._receive_response(8)
        if status == 0:
            max_size = struct.unpack('<Q', response)[0]
            return GetMaxSizeResponse(max_size=max_size)
        raise RPCConnectionError("Failed to get max size")

    def get_base(self, remote_ptr: int) -> BufferGetBaseResponse:
        req = BufferGetBaseRequest(remote_ptr=remote_ptr)
        payload = struct.pack('<Q', req.remote_ptr)
        packed = self._pack_message(RPCCommands.BUFFER_GET_BASE, payload)
        self.socket.send(packed)
        
        status, response = self._receive_response(8)
        if status == 0:
            base_ptr = struct.unpack('<Q', response)[0]
            return BufferGetBaseResponse(base_ptr=base_ptr)
        raise RPCConnectionError("Failed to get base pointer")

    def free_buffer(self, remote_ptr: int) -> None:
        req = FreeBufferRequest(remote_ptr=remote_ptr)
        payload = struct.pack('<Q', req.remote_ptr)
        packed = self._pack_message(RPCCommands.FREE_BUFFER, payload)
        self.socket.send(packed)
        
        status, _ = self._receive_response(0)
        if status != 0:
            raise RPCConnectionError("Failed to free buffer")

    def buffer_clear(self, remote_ptr: int, value: int) -> None:
        req = BufferClearRequest(remote_ptr=remote_ptr, value=value)
        payload = struct.pack('<QB', req.remote_ptr, req.value)
        packed = self._pack_message(RPCCommands.BUFFER_CLEAR, payload)
        self.socket.send(packed)
        
        status, _ = self._receive_response(0)
        if status != 0:
            raise RPCConnectionError("Failed to clear buffer")

    def set_tensor(self, tensor: RPCTensor) -> None:
        payload = TensorBuilder.pack_tensor(tensor)
        packed = self._pack_message(RPCCommands.SET_TENSOR, payload)
        self.socket.send(packed)
        
        status, _ = self._receive_response(0)
        if status != 0:
            raise RPCConnectionError("Failed to set tensor")

    def get_tensor(self, tensor: RPCTensor, offset: int, size: int) -> bytes:
        req = GetTensorRequest(tensor=tensor, offset=offset, size=size)
        payload = TensorBuilder.pack_tensor(req.tensor) + struct.pack('<QQ', recv[0x8:0x10])[0]
        raise RPCConnectionError("Failed to get base pointer")

    def copy_tensor(self, tensor_src: bytes, tensor_dst: bytes) -> None:
        payload = tensor_src + tensor_dst
        packed = self._pack_message(RPCCommands.COPY_TENSOR, payload)
        self.socket.send(packed) 