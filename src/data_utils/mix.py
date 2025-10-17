from typing import List
from .loaders import map_cnn_dailymail, map_xsum, map_hotpotqa
from .schema import Example

def build_mixture(dsets: List[str], split: str = "validation[:200]") -> List[Example]:
    out: List[Example] = []
    for name in dsets:
        if name == "cnn_dailymail":
            out += map_cnn_dailymail(split)
        elif name == "xsum":
            out += map_xsum(split)
        elif name == "hotpotqa":
            out += map_hotpotqa(split)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
    return out
