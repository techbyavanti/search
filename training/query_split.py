"""Shared, deterministic train/held-out query-id split.

Used by every training script under training/ so all of them hold out the
*same* WANDS queries — a model trained for one track never gets evaluated
on a query another track trained on, and the held-out set is stable across
retraining runs.
"""
import random

TRAIN_FRACTION = 0.8
SPLIT_SEED = 42


def split_query_ids(query_ids):
    """Returns (train_query_ids, held_out_query_ids), deterministically shuffled."""
    ids = sorted(query_ids)
    random.Random(SPLIT_SEED).shuffle(ids)
    split_index = int(len(ids) * TRAIN_FRACTION)
    return ids[:split_index], ids[split_index:]
