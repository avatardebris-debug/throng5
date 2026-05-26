"""Add ensure_throng_paths() to repo-root scripts importing throng4."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"bootstrap_paths.py", "tools/split_montezuma_modes.py"}

BOOTSTRAP = (
    "from bootstrap_paths import ensure_throng_paths\n\n"
    "ensure_throng_paths()\n\n"
)

for path in sorted(ROOT.glob("*.py")):
    if path.name in SKIP or path.name.startswith("test_"):
        continue
    text = path.read_text(encoding="utf-8")
    if "from throng4" not in text and "import throng4" not in text:
        continue
    if "ensure_throng_paths" in text:
        continue
    if "from __future__ import annotations" in text:
        parts = text.split("from __future__ import annotations\n", 1)
        if len(parts) == 2:
            # insert after future import block (may include blank line)
            rest = parts[1]
            if rest.startswith("\n"):
                text = parts[0] + "from __future__ import annotations\n" + BOOTSTRAP + rest.lstrip("\n")
            else:
                text = parts[0] + "from __future__ import annotations\n\n" + BOOTSTRAP + rest
    else:
        text = BOOTSTRAP + text
    path.write_text(text, encoding="utf-8")
    print("patched", path.name)
