"""Lead map: each wired input's wire colour, site position and contact."""
from pieeg_server import acq_viewer as av
from pieeg_server.scope_console import (_IRONBCI32_ELECTRODES, _lead_colours,
                                        _pg_keys)


def test_ironbci_colours_follow_the_header_strip():
    c = _lead_colours("ironbci32", 32)
    sites = dict(zip(_IRONBCI32_ELECTRODES[:8], c[:8]))
    assert sites == {"Fp1": "yellow", "Fp2": "orange", "Fz": "red",
                     "Pz": "brown", "C3": "green", "C4": "blue",
                     "O1": "purple", "O2": "grey"}
    assert c[8:] == [None] * 24             # only the cap (bank 1) is mapped


def test_pieeg_colours():
    assert _lead_colours("pieeg8", 8) == ["grey", "purple", "blue", "green",
                                          "yellow", "orange", "red", "brown"]


def test_items_skip_unwired_and_place_sites():
    m = av.ViewerModel(32, 512, _IRONBCI32_ELECTRODES, store=None,
                       signal_contact_inputs=32)
    items = av.lead_map_items(m, _lead_colours("ironbci32", 32))
    assert [i["site"] for i in items] == _IRONBCI32_ELECTRODES[:8]
    assert all(i["xy"] == av.HEAD_XY[i["site"]] for i in items)
    assert items[0]["contact"] is None      # no estimate yet
    assert all(c in av.WIRE_HEX for c in _lead_colours("ironbci32", 8))


def test_polygraphy_leads_are_off_the_head():
    pg = _pg_keys(8)
    eeg = _IRONBCI32_ELECTRODES[:8]
    m = av.ViewerModel(16, 512, eeg + pg, store=None,
                       input_labels={k: f"E{i}" for i, k in
                                     enumerate(pg, start=1)},
                       boards=[("IronBCI-32", eeg), ("PiEEG-8", pg)],
                       extra_rows=[("X1", "X2", "EKG")])
    items = av.lead_map_items(
        m, _lead_colours("ironbci32", 8) + _lead_colours("pieeg8", 8))
    x1 = next(i for i in items if i["site"] == "X1")
    assert x1["xy"] is None and x1["colour"] == "grey"
    assert x1["board"] == "PiEEG-8" and x1["input"] == "E1"


def test_bubbles_follow_the_signal_contact():
    import numpy as np
    m = av.ViewerModel(8, 512, _IRONBCI32_ELECTRODES[:8], store=None,
                       signal_contact_inputs=8)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10, (3 * 512, 8))
    x[:, 6] = 0.0                           # O1 reads nothing: off
    m.push(x)
    for _ in range(av.SIGNAL_CONTACT_HOLD):
        assert m.update_signal_contact(full_scale_uv=300000)
    got = {i["site"]: i["contact"] for i in av.lead_map_items(
        m, _lead_colours("ironbci32", 8))}
    assert got["O1"] == "red"
    assert all(v == "green" for k, v in got.items() if k != "O1")
