"""Only the leads chosen on screen are tested: the console's helpers."""
from pieeg_server.scope_console import _untested


def _res():
    return {"first_input": 1, "problem": None,
            "leads": [{"name": f"E{i + 1}", "ohms": 5000.0, "status": "ok",
                       "text": "5.0 kΩ"} for i in range(4)]}


def test_unchosen_leads_are_marked_untested():
    r = _untested(_res(), {0, 2})
    assert [x["status"] for x in r["leads"]] == ["ok", "untested", "ok",
                                                 "untested"]
    assert r["leads"][1]["ohms"] is None and r["leads"][1]["text"] == ""


def test_no_selection_leaves_the_result_alone():
    res = _res()
    assert _untested(res, None) is res


def test_average_ignores_the_unchosen():
    from pieeg_server.acq_viewer import average_impedance
    r = _untested(_res(), {0})
    assert average_impedance(r, [1, 2, 3, 4]) == (5000.0, 0)
