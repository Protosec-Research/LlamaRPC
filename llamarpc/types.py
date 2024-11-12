from dataclasses import dataclass
from typing import List
from enum import IntEnum

GGML_MAX_DIMS = 4
GGML_MAX_OP_PARAMS = 32
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64

class TensorType(IntEnum):
    Q4_0 = 2
    # Add other tensor types as needed

@dataclass
class RPCTensor:
    id: int          # uint64_t
    type: int        # uint32_t
    buffer: int      # uint64_t
    ne: List[int]    # uint32_t[GGML_MAX_DIMS]
    nb: List[int]    # uint32_t[GGML_MAX_DIMS]
    op: int          # uint32_t
    op_params: List[int]  # int32_t[GGML_MAX_OP_PARAMS]
    flags: int       # int32_t
    src: List[int]   # uint64_t[GGML_MAX_SRC]
    view_src: int    # uint64_t
    view_offs: int   # uint64_t
    data: int        # uint64_t
    name: str        # char[GGML_MAX_NAME]

@dataclass
class AllocBufferRequest:
    size: int        # uint64_t

@dataclass
class AllocBufferResponse:
    remote_ptr: int  # uint64_t
    remote_size: int # uint64_t

@dataclass
class GetAlignmentResponse:
    alignment: int   # uint64_t

@dataclass
class GetMaxSizeResponse:
    max_size: int    # uint64_t

@dataclass
class BufferGetBaseRequest:
    remote_ptr: int  # uint64_t

@dataclass
class BufferGetBaseResponse:
    base_ptr: int    # uint64_t

@dataclass
class FreeBufferRequest:
    remote_ptr: int  # uint64_t

@dataclass
class BufferClearRequest:
    remote_ptr: int  # uint64_t
    value: int       # uint8_t

@dataclass
class GetTensorRequest:
    tensor: RPCTensor
    offset: int      # uint64_t
    size: int        # uint64_t

@dataclass
class CopyTensorRequest:
    src: RPCTensor
    dst: RPCTensor

@dataclass
class CopyTensorResponse:
    result: int      # uint8_t

@dataclass
class GraphComputeResponse:
    result: int      # uint8_t

@dataclass
class GetDeviceMemoryResponse:
    free_mem: int    # uint64_t
    total_mem: int   # uint64_t