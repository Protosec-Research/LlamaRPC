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
from .logger import logger
from ctypes import c_uint64, c_uint32, c_int32, c_uint8

class LlamaRPCConnection:
    def __init__(self, host: str, port: int):
        logger.info(f"Connecting to {host}:{port}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        logger.info("Successfully connected to server")
    
    def _pack_message(self, cmd: RPCCommands, payload: bytes) -> bytes:
        msg_len = len(payload)
        header = bytes([cmd.value]) + struct.pack('<Q', msg_len)
        packed = header + payload
        
        logger.debug(f"Sending command: {cmd.name} ({cmd.value})")
        logger.debug(f"hex    {' '.join(f'{b:02x}' for b in packed)}")
        logger.debug(f"ascii  {''.join(chr(b) if 32 <= b <= 126 else '.' for b in packed)}")
        
        return packed

    def _receive(self, expected_size: int) -> bytes:
        data = bytearray()
        logger.debug(f"Attempting to receive {expected_size} bytes")
        
        while len(data) < expected_size:
            chunk = self.socket.recv(expected_size - len(data))
            if not chunk:
                raise RPCConnectionError("Connection closed by remote host")
            
            logger.debug(f"Received chunk ({len(chunk)} bytes):")
            logger.debug(f"hex    {' '.join(f'{b:02x}' for b in chunk)}")
            logger.debug(f"ascii  {''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)}")
            
            data.extend(chunk)
        
        logger.debug(f"Complete received data ({len(data)} bytes):")
        logger.debug(f"hex    {' '.join(f'{b:02x}' for b in data)}")
        logger.debug(f"ascii  {''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)}")
        
        return bytes(data)

    def _decode_response(self, response_type: Any, data: bytes) -> Any:
        """
        Decode response data based on the response type class
        """
        result = {}
        offset = 0
        
        # Get field types from dataclass annotations
        for field_name, field_type in response_type.__annotations__.items():
            if offset >= len(data):
                logger.warning(f"Insufficient data for field {field_name}")
                break
            
            if field_type == c_uint64:
                if offset + 8 <= len(data):
                    value = struct.unpack('<Q', data[offset:offset + 8])[0]
                    offset += 8
                else:
                    logger.warning(f"Insufficient data for c_uint64 field {field_name}")
                    break
            elif field_type == c_uint32:
                if offset + 4 <= len(data):
                    value = struct.unpack('<I', data[offset:offset + 4])[0]
                    offset += 4
                else:
                    logger.warning(f"Insufficient data for c_uint32 field {field_name}")
                    break
            elif field_type == c_uint8:
                if offset + 1 <= len(data):
                    value = struct.unpack('<B', data[offset:offset + 1])[0]
                    offset += 1
                else:
                    logger.warning(f"Insufficient data for c_uint8 field {field_name}")
                    break
            elif field_type == c_int32:
                if offset + 4 <= len(data):
                    value = struct.unpack('<i', data[offset:offset + 4])[0]
                    offset += 4
                else:
                    logger.warning(f"Insufficient data for c_int32 field {field_name}")
                    break
            else:
                raise ValueError(f"Unsupported type: {field_type}")
            
            result[field_name] = value
        
        return response_type(**result)

    def _receive_response(self, response_type: Optional[Any] = None) -> Tuple[int, Any]:
        # First receive the header (8 bytes) which contains total size
        header = self._receive(8)
        total_size = struct.unpack('<Q', header)[0]
        logger.debug(f"Response header - total size: {total_size}")
        
        # Receive the entire payload
        payload = self._receive(total_size)
        if response_type:
            return total_size, self._decode_response(response_type, payload)
        return total_size, payload

    def alloc_buffer(self, size: int) -> Optional[int]:
        req = AllocBufferRequest(size=size)
        payload = struct.pack('<Q', req.size)
        packed = self._pack_message(RPCCommands.ALLOC_BUFFER, payload)
        self.socket.send(packed)
        
        _, response = self._receive_response(AllocBufferResponse)
        if response and hasattr(response, 'remote_ptr'):
            logger.debug(f"Allocated buffer at: {hex(response.remote_ptr)}, size: {getattr(response, 'remote_size', 'unknown')}")
            return response.remote_ptr
        logger.error("Failed to get buffer pointer")
        return None

    def get_alignment(self) -> GetAlignmentResponse:
        packed = self._pack_message(RPCCommands.GET_ALIGNMENT, b'')
        self.socket.send(packed)
        
        _, response = self._receive_response(GetAlignmentResponse)
        if response:
            return response
        raise RPCConnectionError("Failed to get alignment")

    def get_max_size(self) -> GetMaxSizeResponse:
        packed = self._pack_message(RPCCommands.GET_MAX_SIZE, b'')
        self.socket.send(packed)
        
        _, response = self._receive_response(GetMaxSizeResponse)
        if response:
            return response
        raise RPCConnectionError("Failed to get max size")

    def get_base(self, remote_ptr: int) -> BufferGetBaseResponse:
        req = BufferGetBaseRequest(remote_ptr=remote_ptr)
        payload = struct.pack('<Q', req.remote_ptr)
        packed = self._pack_message(RPCCommands.BUFFER_GET_BASE, payload)
        self.socket.send(packed)
        
        _, response = self._receive_response(BufferGetBaseResponse)
        if response:
            return response
        raise RPCConnectionError("Failed to get base pointer")

    def free_buffer(self, remote_ptr: int) -> None:
        req = FreeBufferRequest(remote_ptr=remote_ptr)
        payload = struct.pack('<Q', req.remote_ptr)
        packed = self._pack_message(RPCCommands.FREE_BUFFER, payload)
        self.socket.send(packed)
        
        total_size, _ = self._receive_response()
        if total_size <= 8:
            raise RPCConnectionError("Failed to free buffer")

    def buffer_clear(self, remote_ptr: int, value: int) -> None:
        req = BufferClearRequest(remote_ptr=remote_ptr, value=value)
        payload = struct.pack('<QB', req.remote_ptr, req.value)
        packed = self._pack_message(RPCCommands.BUFFER_CLEAR, payload)
        self.socket.send(packed)
        
        total_size, _ = self._receive_response()
        if total_size <= 8:
            raise RPCConnectionError("Failed to clear buffer")

    def set_tensor(self, tensor: RPCTensor) -> None:
        payload = TensorBuilder.pack_tensor(tensor)
        packed = self._pack_message(RPCCommands.SET_TENSOR, payload)
        self.socket.send(packed)
        
        total_size, _ = self._receive_response()
        if total_size <= 8:
            raise RPCConnectionError("Failed to set tensor")

    def get_tensor(self, tensor: RPCTensor, offset: int, size: int) -> bytes:
        req = GetTensorRequest(tensor=tensor, offset=offset, size=size)
        payload = TensorBuilder.pack_tensor(req.tensor) + struct.pack('<QQ', req.offset, req.size)
        packed = self._pack_message(RPCCommands.GET_TENSOR, payload)
        self.socket.send(packed)
        
        _, response = self._receive_response()
        if response:
            return response
        raise RPCConnectionError("Failed to get tensor data")

    def copy_tensor(self, tensor_src: RPCTensor, tensor_dst: RPCTensor) -> None:
        req = CopyTensorRequest(src=tensor_src, dst=tensor_dst)
        payload = TensorBuilder.pack_tensor(req.src) + TensorBuilder.pack_tensor(req.dst)
        packed = self._pack_message(RPCCommands.COPY_TENSOR, payload)
        self.socket.send(packed)
        
        total_size, _ = self._receive_response()
        if total_size <= 8:
            raise RPCConnectionError("Failed to copy tensor")

    def graph_compute(self, tensor: RPCTensor) -> GraphComputeResponse:
        payload = TensorBuilder.pack_tensor(tensor)
        packed = self._pack_message(RPCCommands.GRAPH_COMPUTE, payload)
        self.socket.send(packed)
        
        _, response = self._receive_response(GraphComputeResponse)
        if response:
            return response
        raise RPCConnectionError("Failed to compute graph")

    def get_device_memory(self) -> GetDeviceMemoryResponse:
        packed = self._pack_message(RPCCommands.GET_DEVICE_MEMORY, b'')
        self.socket.send(packed)
        
        _, response = self._receive_response(GetDeviceMemoryResponse)
        if response:
            return response
        raise RPCConnectionError("Failed to get device memory")

    def close(self) -> None:
        """Close the socket connection"""
        if hasattr(self, 'socket'):
            logger.info("Closing socket connection")
            self.socket.close()
            logger.info("Socket connection closed")