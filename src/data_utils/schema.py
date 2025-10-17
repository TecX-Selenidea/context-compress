from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class Example:
    source_id: str
    input_text: str
    targets: Dict[str, Optional[str]]   # {"summary":..., "question":..., "answer":...}
    aux: Dict[str, Optional[str]]       # e.g., {"title": "...", "dataset": "..."}
