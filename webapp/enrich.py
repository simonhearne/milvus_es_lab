"""Display fields for result cards, sourced from the parquet.

Not from the engines, deliberately. Three reasons: it puts zero extra load on
the engines whose latency we're measuring; the cards still render when an engine
is down; and it keeps payload out of the timed call, which is why es_engine.py
passes _source=False in the first place.

Hydration happens AFTER the timed runs. It is never inside the timing loop.
"""
import os
import sys
import json

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PARQUET = os.path.join(ROOT, "data", "amazon_reviews.parquet")
SCHEMA = os.path.join(ROOT, "data", "schema.json")

sys.path.insert(0, os.path.join(ROOT, "data"))
from fetch import require_parquet

DISPLAY_COLS = ["title", "store", "price", "average_rating",
                "main_category", "image_url"]


class DisplayIndex:
    def __init__(self, parquet=PARQUET, schema=SCHEMA):
        meta = json.load(open(schema))
        pk = meta["primary_key"]
        # columns= is load-bearing: without it pandas also reads text_vec and
        # image_vec -- ~800MB of vectors held to show product titles.
        df = pd.read_parquet(require_parquet(parquet), columns=[pk] + DISPLAY_COLS)
        df[pk] = df[pk].astype(str)
        self.pk = pk
        self._by_id = df.set_index(pk).to_dict(orient="index")

    @property
    def size(self):
        return len(self._by_id)

    def get(self, pk):
        row = self._by_id.get(str(pk))
        if row is None:
            return None
        return {k: (None if pd.isna(v) else v) for k, v in row.items()}

    def many(self, ids):
        """Hydrate what we can. A miss is omitted, not an error -- the card
        falls back to rendering the id alone."""
        out = {}
        for i in ids:
            if i is None:
                continue
            row = self.get(i)
            if row is not None:
                out[str(i)] = row
        return out
