"""Нагрузка (шаг 2.3): факты, оценки, проверка согласованности. Без сети."""
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "network_sources"
sys.path.insert(0, str(ROOT / "validation"))


def _load(name):
    return yaml.safe_load((SRC / name).read_text(encoding="utf-8"))


def test_load_facts_format():
    d = _load("load_facts.yaml")
    req = {"object", "field", "value", "unit", "source", "accessed", "location", "quote", "confidence"}
    for f in d["facts"]:
        assert req <= set(f), f["object"]
        assert str(f["source"]).startswith("http")
        assert f["confidence"] in ("прямо указано", "выведено")
        if f["confidence"] == "выведено":
            assert f.get("note"), f["object"]  # у вывода должно быть обоснование
    assert d["not_found"]


def test_consumer_shares_sum_to_100():
    d = _load("load_facts.yaml")
    shares = next(f for f in d["facts"] if f["field"] == "доли потребителей по годам")["value"]
    for year, v in shares.items():
        assert sum(v.values()) == pytest.approx(100, abs=0.05), year


def test_estimates_marked_and_allocated():
    e = _load("load_estimates.yaml")
    for c in e["consumers"]:
        assert c["status"] in ("источник", "допущение")
        if c["status"] == "источник":
            assert c["source"].startswith("http") and "energy_mln_kwh" in c
        else:
            assert c.get("basis") and ("avg_mw" in c or "energy_mln_kwh" in c)
        assert sum(a["frac"] for a in c["allocation"]) == pytest.approx(1, abs=1e-3)
        assert all(a["status"] == "допущение" and a["basis"] for a in c["allocation"])
    for s in e["scenarios"].values():
        assert s["status"] == "допущение" and s["basis"]


@pytest.mark.skipif(not shutil.which("node"), reason="нет Node.js для синтетической истории day.js")
def test_check_load_passes():
    import check_load
    assert check_load.main() == 0
