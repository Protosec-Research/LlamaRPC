from enum import IntEnum

class RPCCommands(IntEnum):
    ALLOC_BUFFER = 0
    GET_ALIGNMENT = 1
    GET_MAX_SIZE = 2
    BUFFER_GET_BASE = 3
    FREE_BUFFER = 4
    BUFFER_CLEAR = 5
    SET_TENSOR = 6
    GET_TENSOR = 7
    COPY_TENSOR = 8
    GRAPH_COMPUTE = 9
    GET_DEVICE_MEMORY = 10
    COUNT = 11 