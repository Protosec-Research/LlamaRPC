from llamarpc.connection import LlamaRPCConnection
from llamarpc.types import RPCTensor, TensorType
from llamarpc.logger import logger

def main():
    conn = None
    try:
        logger.info("Initializing RPC connection")
        conn = LlamaRPCConnection("127.0.0.1", 50052)
        
        logger.info("Testing buffer allocation")
        buffer_size = 0xaa
        buffer = conn.alloc_buffer(buffer_size)
        if buffer is None:
            logger.error("Failed to allocate buffer")
            return
        logger.info(f"Successfully allocated buffer at {hex(buffer)}")
    except Exception as e:
        logger.exception("Error occurred")
    finally:
        if conn:
            conn.close()
            logger.info("Connection closed")

if __name__ == "__main__":
    main()
