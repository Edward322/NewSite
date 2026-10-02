import numpy as np
import pandas as pd

from bot.data.bars import aggregate
from tests.helpers import MIN, T0, random_walk


def test_aggregate_matches_pandas_resample():
    md = random_walk(24 * 60, seed=3)
    b = aggregate(md, "1h")
    df = pd.DataFrame({"o": md.open, "h": md.high, "l": md.low, "c": md.close, "v": md.volume},
                      index=pd.to_datetime(md.ts, unit="ms", utc=True))
    r = df.resample("1h").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    assert len(b) == 24
    np.testing.assert_allclose(b.open, r.o); np.testing.assert_allclose(b.high, r.h)
    np.testing.assert_allclose(b.low, r.l); np.testing.assert_allclose(b.close, r.c)
    np.testing.assert_allclose(b.volume, r.v)
    assert (b.close_ts - b.ts == 3_600_000).all()


def test_partial_edge_bars_are_dropped():
    md = random_walk(24 * 60, seed=1).slice(T0 + 7 * MIN, T0 + 24 * 60 * MIN - 5 * MIN)
    b = aggregate(md, "1h")
    assert b.ts[0] == T0 + 3_600_000 and b.ts[-1] == T0 + 22 * 3_600_000


def test_bar_with_missing_minute_is_dropped_and_does_not_leak_into_neighbours():
    md = random_walk(3 * 60, seed=2)
    keep = np.ones(len(md), bool); keep[70] = False          # дыра во второй свече
    from bot.data.bars import MinuteData
    md2 = MinuteData(**{k: v[keep] for k, v in md.__dict__.items()})
    b = aggregate(md2, "1h")
    assert list(b.ts) == [T0, T0 + 2 * 3_600_000]
    assert b.high[1] == md.high[120:180].max()                # третья свеча не захватила минуты второй
