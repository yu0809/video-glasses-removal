"""Small patch for ProPainter: force CPU when the environment variable PROPAINTER_FORCE_CPU=1 is set. Safe to run repeatedly."""
import os
import sys

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "ProPainter", "model", "misc.py")
src = open(path, encoding="utf-8").read()
marker = "PROPAINTER_FORCE_CPU"
if marker in src:
    print("ProPainter is already patched")
    sys.exit(0)
old = "    if IS_HIGH_VERSION:\n        if torch.backends.mps.is_available():"
new = ("    if os.environ.get('PROPAINTER_FORCE_CPU'):\n        return torch.device('cpu')\n" + old)
assert old in src, "ProPainter source layout changed; patch not applied"
open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
print("ProPainter patch applied")
