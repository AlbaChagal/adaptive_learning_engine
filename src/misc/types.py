from typing import Union, Dict, List, Type

NumberType: Type = Union[int, float]
DataSetType: Type = List[Dict[str, Union[str, NumberType, Dict[str, float]]]]
ContextRowType: Type = Dict[str, Union[NumberType, Dict[str, NumberType]]]