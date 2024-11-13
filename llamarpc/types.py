from dataclasses import dataclass
from typing import List
from enum import IntEnum
from ctypes import c_uint64, c_uint32, c_int32, c_uint8, c_char

# Constants
GGML_MAX_DIMS = 4
GGML_MAX_OP_PARAMS = 32
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64

# Base types
class TensorType(IntEnum):
    Q4_0 = 2
    # Add other tensor types as needed

@dataclass
class RPCTensor:
    id: c_uint64
    type: c_uint32
    buffer: c_uint64
    ne: List[c_uint32]    # uint32_t[GGML_MAX_DIMS]
    nb: List[c_uint32]    # uint32_t[GGML_MAX_DIMS]
    op: c_uint32
    op_params: List[c_int32]  # int32_t[GGML_MAX_OP_PARAMS]
    flags: c_int32
    src: List[c_uint64]   # uint64_t[GGML_MAX_SRC]
    view_src: c_uint64
    view_offs: c_uint64
    data: c_uint64
    name: str        # char[GGML_MAX_NAME]


@dataclass
class RPCRequest:
    pass

@dataclass
class RPCResponse:
    pass

@dataclass
class AllocBufferRequest(RPCRequest):
    size: c_uint64

@dataclass
class GetMaxSizeResponse(RPCResponse):
    max_size: c_uint64

@dataclass
class BufferGetBaseResponse(RPCResponse):
    base_ptr: c_uint64

@dataclass
class CopyTensorResponse(RPCResponse):
    result: c_uint8

@dataclass
class GraphComputeResponse(RPCResponse):
    result: c_uint8

@dataclass
class GetDeviceMemoryResponse(RPCResponse):
    free_mem: c_uint64
    total_mem: c_uint64


# Requests
@dataclass
class FreeBufferRequest(RPCRequest):
    remote_ptr: c_uint64

@dataclass
class BufferClearRequest(RPCRequest):
    remote_ptr: c_uint64
    value: c_uint8

@dataclass
class GetTensorRequest(RPCRequest):
    tensor: RPCTensor
    offset: c_uint64
    size: c_uint64

@dataclass
class BufferGetBaseRequest(RPCRequest):
    remote_ptr: c_uint64

@dataclass
class CopyTensorRequest(RPCRequest):
    src: RPCTensor
    dst: RPCTensor


# All Response classes
@dataclass
class AllocBufferResponse(RPCResponse):
    remote_ptr: c_uint64
    remote_size: c_uint64

@dataclass
class GetAlignmentResponse(RPCResponse):
    alignment: c_uint64

@dataclass
class GetMaxSizeResponse(RPCResponse):
    max_size: c_uint64

@dataclass
class BufferGetBaseResponse(RPCResponse):
    base_ptr: c_uint64

@dataclass
class BasicResponse(RPCResponse):
    success: c_uint8

@dataclass
class CopyTensorResponse(RPCResponse):
    result: c_uint8

@dataclass
class GraphComputeResponse(RPCResponse):
    result: c_uint8

@dataclass
class GetDeviceMemoryResponse(RPCResponse):
    free_mem: c_uint64
    total_mem: c_uint64