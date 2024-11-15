# LlamaRPC

A Python RPC client library for interacting with LLaMA servers. This library provides a robust interface for tensor operations, buffer management, and device memory handling.

## Features

- 🔌 TCP socket-based RPC communication
- 📦 Buffer allocation and management
- 🧮 Tensor operations (get, set, copy)
- 📊 Graph computation support
- 💾 Device memory information
- 🪵 Rich logging with detailed debugging

## Installation 

```bash
pip install llamarpc
```

## API Reference

### Connection Management

- `LlamaRPCConnection(host: str, port: int)`: Create a new connection to LLaMA server
- `close()`: Close the connection

### Buffer Operations

- `alloc_buffer(size: int) -> int`: Allocate a buffer of specified size
- `free_buffer(remote_ptr: int)`: Free an allocated buffer
- `buffer_clear(remote_ptr: int, value: int)`: Clear buffer with specified value
- `get_base(remote_ptr: int) -> BufferGetBaseResponse`: Get base pointer of buffer

### Tensor Operations

- `set_tensor(tensor: RPCTensor)`: Set tensor data
- `get_tensor(tensor: RPCTensor, offset: int, size: int) -> bytes`: Get tensor data
- `copy_tensor(tensor_src: RPCTensor, tensor_dst: RPCTensor)`: Copy tensor data
- `graph_compute(tensor: RPCTensor) -> GraphComputeResponse`: Perform graph computation

### Device Information

- `get_alignment() -> GetAlignmentResponse`: Get memory alignment
- `get_max_size() -> GetMaxSizeResponse`: Get maximum allocation size
- `get_device_memory() -> GetDeviceMemoryResponse`: Get device memory information

## Error Handling

The library uses custom `RPCConnectionError` for handling connection-related errors: