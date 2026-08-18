"""
Common interface so the notebook can call Milvus and ES with identical arguments
and diff the results / compute recall without special-casing either engine.

Every search returns a list of (id: str, score: float), highest score first.
"""
from abc import ABC, abstractmethod
from typing import Optional
import time


class Result(list):
    """List of (id, score) with a millisecond timing attached."""
    def __init__(self, hits, ms):
        super().__init__(hits)
        self.ms = ms

    @property
    def ids(self):
        return [h[0] for h in self]


def timed(fn):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        hits = fn(*a, **k)
        return Result(hits, (time.perf_counter() - t0) * 1000)
    return wrap


class Engine(ABC):
    name: str

    @abstractmethod
    def dense_search(self, query_vec, k=10, filter=None): ...

    @abstractmethod
    def text_search(self, text, k=10, filter=None): ...

    @abstractmethod
    def hybrid_search(self, query_vec, text, k=10, filter=None): ...

    @abstractmethod
    def grouped_search(self, query_vec, group_field, k=10): ...
