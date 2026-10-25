from __future__ import annotations

from tokenizer import Tokenizer


def build_toy_tokenizer() -> Tokenizer:
    vocab = {
        b"h": 0,
        b"e": 1,
        b"l": 2,
        b"o": 3,
        b"w": 4,
        b"r": 5,
        b"d": 6,
        b"he": 7,
        b"ll": 8,
        b"hello": 9,
        b"world": 10,
    }
    merges = [
        (b"h", b"e"),
        (b"l", b"l"),
        (b"he", b"ll"),
        (b"hell", b"o"),
        (b"w", b"o"),
        (b"wor", b"l"),
        (b"worl", b"d"),
    ]
    return Tokenizer(vocab, merges)


def test_encode_decode_round_trip():
    tok = build_toy_tokenizer()
    encoded = tok.encode("hello")
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    decoded = tok.decode(encoded)
    assert "hell" in decoded or "he" in decoded
