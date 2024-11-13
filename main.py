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
        logger.info(f"Allocated buffer at {hex(buffer)}")
        base = conn.get_base(buffer)
        logger.info(f"Base pointer: {hex(base.base_ptr)}")
        max_size = conn.get_max_size()
        logger.info(f"Max size: {max_size.max_size}")
        alignment = conn.get_alignment()
        logger.info(f"Alignment: {alignment.alignment}")
        # free_buffer = conn.free_buffer(buffer)
        # logger.info(f"Freed buffer at {hex(buffer)}")
        # conn.buffer_clear(buffer, 0x00)

    except Exception as e:
        logger.exception("Error occurred")
    finally:
        if conn:
            conn.close()
            logger.info("Connection closed")

if __name__ == "__main__":
    main()
