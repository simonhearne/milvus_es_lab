"""Shared reader for data/synonyms.txt (Solr-format synonym rules).

Single source of truth for both loaders. Blank lines and `#` comments are
skipped; every other line is one Solr synonym rule passed verbatim to the
engines (Milvus inline `synonyms` array / ES `synonym` filter).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SYNONYMS_FILE = os.path.join(HERE, "synonyms.txt")


def read_synonyms(path=None):
    """Return the list of rule strings, skipping blanks and `#` comments."""
    path = path or SYNONYMS_FILE
    rules = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rules.append(line)
    return rules
