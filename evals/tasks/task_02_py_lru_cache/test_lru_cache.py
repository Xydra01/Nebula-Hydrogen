from __future__ import annotations

from lru_cache import LRUCache


def test_basic_put_get():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)  # evicts key 2
    assert cache.get(2) == -1
    cache.put(4, 4)  # evicts key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4


def test_update_existing_key():
    cache = LRUCache(2)
    cache.put(1, 10)
    cache.put(1, 20)
    assert cache.get(1) == 20
    cache.put(2, 30)
    cache.put(3, 40)  # evicts 1
    assert cache.get(1) == -1
    assert cache.get(2) == 30


def test_capacity_one():
    cache = LRUCache(1)
    cache.put(10, 100)
    assert cache.get(10) == 100
    cache.put(20, 200)
    assert cache.get(10) == -1
    assert cache.get(20) == 200
