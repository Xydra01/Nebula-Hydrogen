"""BPE Tokenizer implementation for refactoring."""

from __future__ import annotations

import re

GPT2_SPLIT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """Byte-pair encoding tokenizer."""

    def __init__(self, vocab: dict[bytes, int], merges: list[tuple[bytes, bytes]]) -> None:
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        """Encode string to token ids."""
        tokens: list[int] = []
        for word in re.findall(r"\w+|[^\w\s]", text):
            word_bytes = word.encode("utf-8")
            parts = [bytes([b]) for b in word_bytes]
            while len(parts) > 1:
                pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
                min_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
                if min_pair not in self.bpe_ranks:
                    break
                new_parts: list[bytes] = []
                i = 0
                while i < len(parts):
                    if (
                        i < len(parts) - 1
                        and parts[i] == min_pair[0]
                        and parts[i + 1] == min_pair[1]
                    ):
                        new_parts.append(min_pair[0] + min_pair[1])
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                parts = new_parts
            for p in parts:
                if p in self.vocab:
                    tokens.append(self.vocab[p])
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to string."""
        raw = b"".join(self.inv_vocab.get(i, b"") for i in ids)
        return raw.decode("utf-8", errors="replace")
