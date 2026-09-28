"""The pixel trace layer: columns match the reference envelope, spans join
neighbouring columns, and only changed columns are handed to Tk."""
import numpy as np

from pieeg_server.acq_viewer import (SweepRaster, column_envelope,
                                     sweep_envelope)


def test_column_envelope_matches_sweep_envelope():
    rng = np.random.default_rng(4)
    win, ncol = 5120, 780
    trace = rng.normal(0, 30, win)
    head = 1234
    vals, _ = sweep_envelope(trace, head, ncol)
    starts = (np.arange(ncol) * win) // ncol
    seg = np.roll(trace, head)[:, None]
    first, second = column_envelope(seg, starts)
    assert np.array_equal(first[0], vals[0::2])
    assert np.array_equal(second[0], vals[1::2])


def test_spans_join_and_rows_keep_their_colour():
    r = SweepRaster(10, 40, "#000000")
    r.clear(rows=2, ncol=10)
    first = np.array([[5.0, 9.0, 7.0], [25.0, 25.0, 30.0]])
    second = np.array([[8.0, 6.0, 7.0], [26.0, 24.0, 30.0]])
    r.draw(2, 4, 10, first, second, ["#0000ff", "#ff0000"])
    col = lambda x, c: set(np.flatnonzero(np.all(r.img[:, x] == c, axis=1)))
    blue, red = (0, 0, 255), (255, 0, 0)
    assert col(2, blue) == set(range(5, 9))          # 5..8
    assert col(3, blue) == set(range(6, 10))         # from 8 (end of col 2) to 9, 6
    assert col(4, blue) == {6, 7}                    # joins 6 -> 7
    assert col(2, red) == {25, 26} and col(4, red) == set(range(24, 31))
    assert col(1, blue) == set() and col(5, blue) == set()


def test_only_changed_columns_are_dirty():
    r = SweepRaster(100, 20, "#000000")
    r.clear(rows=1, ncol=100)
    assert r.take_dirty() == [(0, 100)]
    r.draw(40, 42, 100, np.full((1, 3), 5.0), np.full((1, 3), 6.0),
           ["#ffffff"])
    r.erase(43, 44, 100)
    assert r.take_dirty() == [(40, 45)]
    assert r.take_dirty() == []
    assert r.ppm(40, 45).startswith(b"P6 5 20 255 ")
    assert len(r.ppm(40, 45)) == len(b"P6 5 20 255 ") + 5 * 20 * 3


def test_fewer_columns_than_pixels_fill_every_pixel():
    r = SweepRaster(10, 10, "#000000")
    r.clear(rows=1, ncol=4)
    r.draw(0, 3, 4, np.full((1, 4), 3.0), np.full((1, 4), 3.0), ["#ffffff"])
    assert np.all(r.img[3, :] == 255)
