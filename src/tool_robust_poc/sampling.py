"""Deterministic item ordering utilities for staged experiment runs."""

from __future__ import annotations

import hashlib
import json
from typing import Callable, TypeVar

T = TypeVar("T")


def stable_digest(text: str, *, length: int = 16) -> str:
    """Return a short stable digest for identifiers and cache tags."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def stable_json(value) -> str:
    """Serialize a value deterministically for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_sample(
    items: list[T],
    *,
    stable_key_fn: Callable[[T], str],
    seed: str,
    limit: int | None = None,
    order: str = "hash",
) -> list[dict]:
    """Return items in a deterministic canonical order.

    Each returned record contains the original item plus:
      - `ordered_index`: index after deterministic ordering
      - `original_index`: index in the source file / dataset order
      - `item_key`: stable string key used for ordering
      - `item_id`: short digest derived from the item key
    """
    records = []
    for original_index, item in enumerate(items):
        item_key = stable_key_fn(item)
        item_id = stable_digest(item_key)
        if order == "hash":
            sort_key = stable_digest(f"{seed}:{item_key}", length=64)
        elif order == "file":
            sort_key = f"{original_index:09d}"
        else:
            raise ValueError(f"Unknown sample order: {order!r}")
        records.append(
            {
                "item": item,
                "item_key": item_key,
                "item_id": item_id,
                "original_index": original_index,
                "sort_key": sort_key,
            }
        )

    ordered = sorted(records, key=lambda record: (record["sort_key"], record["item_key"]))
    if limit is not None:
        ordered = ordered[:limit]

    for ordered_index, record in enumerate(ordered):
        record["ordered_index"] = ordered_index

    return ordered
