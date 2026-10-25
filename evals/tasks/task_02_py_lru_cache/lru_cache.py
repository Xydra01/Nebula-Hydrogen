"""LRU Cache starter stub."""

from __future__ import annotations


class LRUCache:
    """Least Recently Used (LRU) Cache."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def get(self, key: int) -> int:
        """Retrieve key value or -1 if not found."""
        raise NotImplementedError

    def put(self, key: int, value: int) -> None:
        """Insert or update key value, evicting LRU item if exceeding capacity."""
        raise NotImplementedError
