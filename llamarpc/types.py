from dataclasses import dataclass
from typing import List
from enum import IntEnum
from ctypes import c_uint64, c_uint32, c_int32, c_uint8, c_char

GGML_MAX_DIMS = 4
GGML_MAX_OP_PARAMS = 32
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64

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
class AllocBufferRequest:
    size: c_uint64

@dataclass
class AllocBufferResponse:
    remote_ptr: c_uint64
    remote_size: c_uint64

@dataclass
class GetAlignmentResponse:
    alignment: c_uint64

@dataclass
class GetMaxSizeResponse:
    max_size: c_uint64

@dataclass
class BufferGetBaseRequest:
    remote_ptr: c_uint64

@dataclass
class BufferGetBaseResponse:
    base_ptr: c_uint64

@dataclass
class FreeBufferRequest:
    remote_ptr: c_uint64

@dataclass
class BufferClearRequest:
    remote_ptr: c_uint64
    value: c_uint8

@dataclass
class GetTensorRequest:
    tensor: RPCTensor
    offset: c_uint64
    size: c_uint64

@dataclass
class CopyTensorRequest:
    src: RPCTensor
    dst: RPCTensor

@dataclass
class CopyTensorResponse:
    result: c_uint8

@dataclass
class GraphComputeResponse:
    result: c_uint8

@dataclass
class GetDeviceMemoryResponse:
    free_mem: c_uint64
    total_mem: c_uint64