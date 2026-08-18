"""The long-lived execution namespaces the editable snippets run inside.

Each pane runs against its own namespace where `client` is that pane's SDK
client -- MilvusClient in the Milvus pane, Elasticsearch in the ES pane -- so
the side-by-side snippets address their engine by the same name and read as
one diff. The other bindings (`qv`, `img_qv`, `SEARCH`, ...) are shared.
SEARCH is imported from MilvusEngine rather than copied, so tuning nprobe
stays a one-line change in one file.

Query vectors come from gt.npz, not the parquet: the ground truth already stores
them, and reloading a 1024-d vector column costs 400 MB for two floats' worth of
actual need.

Construction tolerates unreachable engines. A dead stack must cost you the
numbers, not the code on screen -- the server still boots and health reports red.
"""
import os
import json
import datetime as dt

import numpy as np
from pymilvus import (MilvusClient, AnnSearchRequest, RRFRanker,
                      Function, FunctionType, FunctionScore, LexicalHighlighter)
from elasticsearch import Elasticsearch

from engines.milvus_engine import MilvusEngine

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCHEMA = os.path.join(ROOT, "data", "schema.json")
GT = os.path.join(ROOT, "bench", "gt.npz")

TIMEOUT = float(os.environ.get("DEMO_TIMEOUT", "30"))


class Session:
    def __init__(self):
        self.meta = json.load(open(SCHEMA))
        self.pk = self.meta["primary_key"]
        self.errors = {"milvus": None, "es": None, "gt": None}

        self.client = None
        try:
            self.client = MilvusClient(
                uri=os.environ.get("MILVUS_URI", "http://localhost:19530"),
                timeout=TIMEOUT,
            )
        except Exception as e:
            self.errors["milvus"] = f"{type(e).__name__}: {e}"

        self.es = None
        try:
            self.es = Elasticsearch(
                os.environ.get("ES_URI", "http://localhost:9200"),
                request_timeout=TIMEOUT,
            )
        except Exception as e:
            self.errors["es"] = f"{type(e).__name__}: {e}"

        self.gt = None
        try:
            self.gt = np.load(GT, allow_pickle=True)
        except Exception as e:
            self.errors["gt"] = f"{type(e).__name__}: {e}"

    # ---- queries ------------------------------------------------------------

    def query_count(self):
        return 0 if self.gt is None else int(self.gt["query_idx"].shape[0])

    def query_row_id(self, i):
        """The parent_asin of the row query i was drawn from."""
        if self.gt is None:
            return None
        return str(self.gt["ids"][int(self.gt["query_idx"][i])])

    def truth_ids(self, space, i):
        """Ground-truth neighbour ids for query i in `space` ('text'|'image')."""
        if self.gt is None:
            return None
        key = "gt" if space == "text" else "gt_image"
        return [str(self.gt["ids"][j]) for j in self.gt[key][i]]

    # ---- namespace ----------------------------------------------------------

    def namespace(self, query_index=0, engine="milvus"):
        """One pane's namespace. `engine` decides what `client` is bound to:
        the MilvusClient ("milvus") or the Elasticsearch client ("es")."""
        ns = {
            "client": self.client if engine == "milvus" else self.es,
            "AnnSearchRequest": AnnSearchRequest,
            "RRFRanker": RRFRanker,
            "SEARCH": MilvusEngine.SEARCH,
            # Milvus 3.0 relevance shaping + highlighting.
            "Function": Function,
            "FunctionType": FunctionType,
            "FunctionScore": FunctionScore,
            "LexicalHighlighter": LexicalHighlighter,
            # The dates panel computes its own boundary: Milvus has no date math.
            "datetime": dt.datetime,
            "timezone": dt.timezone,
            "timedelta": dt.timedelta,
        }
        if self.gt is not None:
            ns["qv"] = self.gt["q_text"][query_index].tolist()
            ns["img_qv"] = self.gt["q_image"][query_index].tolist()
        return ns

    # ---- health -------------------------------------------------------------

    def health(self):
        return {"milvus": self._milvus_health(), "es": self._es_health()}

    def _milvus_health(self):
        if self.client is None:
            return {"ok": False, "error": self.errors["milvus"]}
        try:
            self.client.list_collections()
            return {"ok": True, "error": None}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _es_health(self):
        if self.es is None:
            return {"ok": False, "error": self.errors["es"]}
        try:
            if self.es.ping():
                return {"ok": True, "error": None}
            return {"ok": False, "error": "ping returned False"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
