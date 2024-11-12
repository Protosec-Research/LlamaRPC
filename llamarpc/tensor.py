from typing import List
import struct
from .types import TensorParams, TensorDimensions, TensorType

class TensorBuilder:
    @staticmethod
    def pack_tensor(params: TensorParams) -> bytes:
        tensor_data = bytearray()
        
        # Pack basic fields
        tensor_data.extend(struct.pack('<Q', params.id))
        tensor_data.extend(struct.pack('<I', params.type))
        tensor_data.extend(struct.pack('<Q', params.buffer))
        
        # Pack dimensions
        for n in params.dimensions.ne:
            tensor_data.extend(struct.pack('<I', n))
        for n in params.dimensions.nb:
            tensor_data.extend(struct.pack('<I', n))
            
        # Pack remaining fields
        tensor_data.extend(struct.pack('<I', params.op))
        tensor_data.extend(b''.join(struct.pack('<I', x) for x in params.op_params))
        tensor_data.extend(struct.pack('<I', params.flags))
        tensor_data.extend(b''.join(struct.pack('<Q', x) for x in params.src))
        tensor_data.extend(struct.pack('<Q', params.view_src))
        tensor_data.extend(struct.pack('<Q', params.view_offs))
        tensor_data.extend(struct.pack('<Q', params.data))
        
        # Pack name and padding
        name_bytes = params.name.encode()[:64].ljust(64, b'\x00')
        tensor_data.extend(name_bytes)
        tensor_data.extend(b'\x00' * 4)  # padding
        
        return bytes(tensor_data) 