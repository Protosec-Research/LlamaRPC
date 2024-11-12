from llamarpc.connection import LlamaRPCConnection
from llamarpc.types import RPCTensor, TensorType
import time

def main():
    # Connect to the RPC server with debug enabled
    try:
        conn = LlamaRPCConnection("127.0.0.1", 50052)
        print("Successfully connected to RPC server")

        # Test device memory info

        # Allocate a buffer
        print("\n=== Testing Buffer Allocation ===")
        buffer_size = 1024  # 1MB
        buffer = conn.alloc_buffer(buffer_size)
        print(f"Allocated buffer at {hex(buffer.remote_ptr)} with size {buffer.remote_size}")

        # Create a sample tensor
        print("\n=== Testing Tensor Operations ===")
        tensor = RPCTensor(
            id=1,
            type=TensorType.Q4_0,
            buffer=buffer.remote_ptr,
            ne=[1, 1, 1, 1],
            nb=[4, 4, 4, 4],
            op=0,
            op_params=[0] * 32,
            flags=0,
            src=[0] * 10,
            view_src=0,
            view_offs=0,
            data=0,
            name="test_tensor"
        )

        # Set tensor
        conn.set_tensor(tensor)
        print("Successfully set tensor")

        # Clear buffer
        print("\n=== Testing Buffer Clear ===")
        conn.buffer_clear(buffer.remote_ptr, 0)
        print("Successfully cleared buffer")

        # Free buffer
        print("\n=== Testing Buffer Free ===")
        conn.free_buffer(buffer.remote_ptr)
        print("Successfully freed buffer")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
        print("Connection closed")

if __name__ == "__main__":
    main()
