"""Unit tests for data/synthesize.py (no cluster, no parquet).

Run: docker compose exec -T demo python -m webapp.test_synthesize
"""
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

import synthesize


def test_store_location_is_stable_for_the_same_store():
    a = synthesize.store_location_wkt("Anker")
    b = synthesize.store_location_wkt("Anker")
    assert a == b, (a, b)


def test_store_location_differs_across_stores():
    # Not a guarantee for any given pair, but over many stores the generator
    # must actually spread. A single city for everything would make the radius
    # query meaningless.
    seen = {synthesize.store_location_wkt(f"store-{i}") for i in range(200)}
    assert len(seen) > 5, seen


def test_wkt_is_lon_lat_and_geopoint_is_lat_lon():
    wkt = synthesize.store_location_wkt("Anker")
    pt = synthesize.store_location_geopoint("Anker")
    lon, lat = wkt.removeprefix("POINT (").removesuffix(")").split()
    assert float(lon) == pt["lon"]
    assert float(lat) == pt["lat"]


def test_wkt_parses_as_a_point():
    wkt = synthesize.store_location_wkt("Belkin")
    assert wkt.startswith("POINT (") and wkt.endswith(")")
    assert len(wkt.removeprefix("POINT (").removesuffix(")").split()) == 2


def test_first_seen_is_stable_for_the_same_asin():
    now = dt.datetime(2026, 7, 30, tzinfo=dt.timezone.utc)
    assert synthesize.first_seen("B07CCRYMXQ", now) == \
           synthesize.first_seen("B07CCRYMXQ", now)


def test_first_seen_is_rfc3339_utc():
    now = dt.datetime(2026, 7, 30, tzinfo=dt.timezone.utc)
    s = synthesize.first_seen("B07CCRYMXQ", now)
    parsed = dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    assert parsed.year >= 2024


def test_first_seen_never_lands_in_the_future():
    now = dt.datetime(2026, 7, 30, tzinfo=dt.timezone.utc)
    for i in range(500):
        s = synthesize.first_seen(f"asin-{i}", now)
        parsed = dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc)
        assert parsed <= now, (s, now)


def test_enough_products_land_inside_the_last_30_days():
    # The dates panel filters to now-30d and must return a full top-10 out of
    # a 98k corpus. A thin recent tail would render an empty panel on stage.
    now = dt.datetime(2026, 7, 30, tzinfo=dt.timezone.utc)
    cutoff = now - dt.timedelta(days=30)
    recent = 0
    for i in range(1000):
        s = synthesize.first_seen(f"asin-{i}", now)
        parsed = dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc)
        if parsed > cutoff:
            recent += 1
    assert recent > 100, f"only {recent}/1000 inside the window"


def test_store_location_tolerates_missing_store():
    assert synthesize.store_location_wkt(None).startswith("POINT (")
    assert synthesize.store_location_wkt("").startswith("POINT (")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"OK {len(tests)} synthesize tests passed")
