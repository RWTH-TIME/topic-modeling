from typing import List
from dataclasses import dataclass


@dataclass
class PreprocessedDocument:
    doc_id: str
    tokens: List[str]
