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
from .logger import logger, console
from rich.table import Table
from rich.panel import Panel
from rich import box

class LlamaRPCConnection:
    def __init__(self, host: str, port: int):
        logger.info(f"Connecting to {host}:{port}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        logger.info("Successfully connected to server")
        
    def _pack_message(self, cmd: RPCCommands, payload: bytes) -> bytes:
        msg_len = len(payload)
        header = struct.pack('<BQ', cmd, msg_len)
        packed = header + payload
        
        # Create a table for message details
        table = Table(box=box.ROUNDED)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Command", f"{cmd.name} ({cmd.value})")
        table.add_row("Header", header.hex())
        table.add_row("Payload", payload.hex())
        table.add_row("Total Message", packed.hex())
        
        console.print(Panel(table, title="[bold blue]Outgoing Message", border_style="blue"))
        return packed

    def _receive(self, expected_size: int) -> bytes:
        data = bytearray()
        with console.status(f"[bold blue]Receiving {expected_size} bytes..."):
            while len(data) < expected_size:
                chunk = self.socket.recv(expected_size - len(data))
                if not chunk:
                    raise RPCConnectionError("Connection closed by remote host")
                console.print(f"[dim]Received chunk: {chunk.hex()}")
                data.extend(chunk)
        
        console.print(Panel(
            f"[green]Complete received data: {bytes(data).hex()}",
            title="[bold green]Received Data",
            border_style="green"
        ))
        return bytes(data)

    def _receive_response(self, expected_size: int) -> Tuple[int, bytes]:
        logger.debug(f"Receiving response with expected size: {expected_size}")
        header = self._receive(8)
        status, size = struct.unpack('<II', header)
        logger.debug(f"Response header - status: {status}, size: {size}")
        
        if size > 0:
            payload = self._receive(size)
            logger.debug(f"Response payload: {payload.hex()}")
        else:
            payload = b''
            logger.debug("No payload in response")
        return status, payload

    def alloc_buffer(self, size: int) -> Optional[int]:
        req = AllocBufferRequest(size=size)
        payload = struct.pack('<Q', req.size)
        packed = self._pack_message(RPCCommands.ALLOC_BUFFER, payload)
        self.socket.send(packed)
        
        status, response = self._receive_response(16)
        if status == 0 and len(response) >= 16:
            remote_ptr, remote_size = struct.unpack('<QQ', response)
            logger.debug(f"Allocated buffer at: {hex(remote_ptr)}")
            return remote_ptr
        logger.error("Failed to get buffer pointer")
        return None

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
        payload = TensorBuilder.pack_tensor(req.tensor) + struct.pack('<QQ', req.offset, req.size)
        packed = self._pack_message(RPCCommands.GET_TENSOR, payload)
        self.socket.send(packed)
        
        status, response = self._receive_response(size)
        if status == 0:
            return response
        raise RPCConnectionError("Failed to get tensor data")

    def copy_tensor(self, tensor_src: RPCTensor, tensor_dst: RPCTensor) -> None:
        req = CopyTensorRequest(src=tensor_src, dst=tensor_dst)
        payload = TensorBuilder.pack_tensor(req.src) + TensorBuilder.pack_tensor(req.dst)
        packed = self._pack_message(RPCCommands.COPY_TENSOR, payload)
        self.socket.send(packed)
        
        status, _ = self._receive_response(0)
        if status != 0:
            raise RPCConnectionError("Failed to copy tensor")

    def graph_compute(self, tensor: RPCTensor) -> GraphComputeResponse:
        payload = TensorBuilder.pack_tensor(tensor)
        packed = self._pack_message(RPCCommands.GRAPH_COMPUTE, payload)
        self.socket.send(packed)
        
        status, response = self._receive_response(16)
        if status == 0:
            graph_ptr, graph_size = struct.unpack('<QQ', response)
            return GraphComputeResponse(graph_ptr=graph_ptr, graph_size=graph_size)
        raise RPCConnectionError("Failed to compute graph")

    def get_device_memory(self) -> GetDeviceMemoryResponse:
        packed = self._pack_message(RPCCommands.GET_DEVICE_MEMORY, b'')
        self.socket.send(packed)
        
        status, response = self._receive_response(16)
        if status == 0:
            memory_ptr, memory_size = struct.unpack('<QQ', response)
            return GetDeviceMemoryResponse(memory_ptr=memory_ptr, memory_size=memory_size)
        raise RPCConnectionError("Failed to get device memory")

    def close(self) -> None:
        """Close the socket connection"""
        if hasattr(self, 'socket'):
            logger.info("Closing socket connection")
            self.socket.close()
            logger.info("Socket connection closed")