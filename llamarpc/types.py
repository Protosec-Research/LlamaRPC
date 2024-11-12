from dataclasses import dataclass
from typing import List, Optional
from enum import IntEnum

class TensorType(IntEnum):
    Q4_0 = 2

@dataclass
class TensorDimensions:
    nb: List[int]  # size: 4
    ne: List[int]  # size: 4

@dataclass
class TensorParams:
    id: int
    type: TensorType
    buffer: int  # buffer pointer
    dimensions: TensorDimensions
    op: int
    op_params: List[int]  # size: 16
    flags: int
    src: List[int]  # size: 10
    view_src: int
    view_offs: int
    data: int
    name: str 