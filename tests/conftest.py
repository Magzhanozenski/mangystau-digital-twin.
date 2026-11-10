import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CASES = ROOT / "validation" / "cases.json"


@pytest.fixture(scope="session")
def net():
    from twin import load_network
    return load_network()


@pytest.fixture(scope="session")
def js_data():
    """Сценарии и результаты JS-ядра. Генерируются через Node.js, если файла ещё нет."""
    if not CASES.exists():
        node = shutil.which("node")
        if not node:
            pytest.skip("нет validation/cases.json и не найден Node.js для его генерации")
        subprocess.run([node, "validation/export.js"], cwd=ROOT, check=True, capture_output=True)
    return json.loads(CASES.read_text(encoding="utf-8"))
