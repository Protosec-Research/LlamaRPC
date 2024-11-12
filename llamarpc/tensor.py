import struct
from .types import RPCTensor, GGML_MAX_DIMS, GGML_MAX_OP_PARAMS, GGML_MAX_SRC, GGML_MAX_NAME

class TensorBuilder:
    @staticmethod
    def pack_tensor(tensor: RPCTensor) -> bytes:
        tensor_data = bytearray()
        
        # Pack fields according to rpc_tensor structure
        tensor_data.extend(struct.pack('<Q', tensor.id))
        tensor_data.extend(struct.pack('<I', tensor.type))
        tensor_data.extend(struct.pack('<Q', tensor.buffer))
        
        # Pack ne and nb arrays
        for i in range(GGML_MAX_DIMS):
            tensor_data.extend(struct.pack('<I', tensor.ne[i] if i < len(tensor.ne) else 0))
        for i in range(GGML_MAX_DIMS):
            tensor_data.extend(struct.pack('<I', tensor.nb[i] if i < len(tensor.nb) else 0))
        
        tensor_data.extend(struct.pack('<I', tensor.op))
        
        # Pack op_params
        for i in range(GGML_MAX_OP_PARAMS // 4):  # Divide by 4 because of int32_t size
            tensor_data.extend(struct.pack('<i', tensor.op_params[i] if i < len(tensor.op_params) else 0))
        
        tensor_data.extend(struct.pack('<i', tensor.flags))
        
        # Pack src array
        for i in range(GGML_MAX_SRC):
            tensor_data.extend(struct.pack('<Q', tensor.src[i] if i < len(tensor.src) else 0))
        
        tensor_data.extend(struct.pack('<Q', tensor.view_src))
        tensor_data.extend(struct.pack('<Q', tensor.view_offs))
        tensor_data.extend(struct.pack('<Q', tensor.data))
        
        # Pack name with padding
        name_bytes = tensor.name.encode()[:GGML_MAX_NAME].ljust(GGML_MAX_NAME, b'\x00')
        tensor_data.extend(name_bytes)
        
        # Add padding to ensure size is multiple of 8
        tensor_data.extend(b'\x00' * 4)
        
        return bytes(tensor_data)

    @staticmethod
    def unpack_tensor(data: bytes) -> RPCTensor:
        offset = 0
        
        id = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        type = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        buffer = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        ne = []
        nb = []
        for _ in range(GGML_MAX_DIMS):
            ne.append(struct.unpack('<I', data[offset:offset+4])[0])
            offset += 4
        for _ in range(GGML_MAX_DIMS):
            nb.append(struct.unpack('<I', data[offset:offset+4])[0])
            offset += 4
        
        op = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        op_params = []
        for _ in range(GGML_MAX_OP_PARAMS // 4):
            op_params.append(struct.unpack('<i', data[offset:offset+4])[0])
            offset += 4
        
        flags = struct.unpack('<i', data[offset:offset+4])[0]
        offset += 4
        
        src = []
        for _ in range(GGML_MAX_SRC):
            src.append(struct.unpack('<Q', data[offset:offset+8])[0])
            offset += 8
        
        view_src = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        view_offs = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        data_ptr = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        
        name = data[offset:offset+GGML_MAX_NAME].decode().rstrip('\x00')
        
        return RPCTensor(
            id=id,
            type=type,
            buffer=buffer,
            ne=ne,
            nb=nb,
            op=op,
            op_params=op_params,
            flags=flags,
            src=src,
            view_src=view_src,
            view_offs=view_offs,
            data=data_ptr,
            name=name
        )