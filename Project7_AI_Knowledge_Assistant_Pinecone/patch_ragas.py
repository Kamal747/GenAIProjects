"""
patch_ragas.py
Fixes ragas's broken unconditional VertexAI import WITHOUT importing ragas
itself (importing it would crash before we get a chance to patch it).

Locates ragas/llms/base.py directly inside the active venv's site-packages
using sys.path / sys.prefix, so no import of the broken package is needed.

Run with:
    python -m patch_ragas
or:
    py patch_ragas.py
"""
import sys
import os
import glob

# Find site-packages for the CURRENTLY ACTIVE interpreter (your .venv)
site_packages_candidates = [p for p in sys.path if p.rstrip("\\/").endswith("site-packages")]

base_file = None
for sp in site_packages_candidates:
    candidate = os.path.join(sp, "ragas", "llms", "base.py")
    if os.path.isfile(candidate):
        base_file = candidate
        break

# Fallback: glob search under sys.prefix (covers unusual venv layouts)
if base_file is None:
    matches = glob.glob(os.path.join(sys.prefix, "**", "ragas", "llms", "base.py"), recursive=True)
    if matches:
        base_file = matches[0]

if base_file is None:
    print("Could not locate ragas/llms/base.py automatically.")
    print("Make sure your .venv is activated and ragas is installed (pip show ragas), then retry.")
    sys.exit(1)

print(f"Found: {base_file}")

with open(base_file, "r", encoding="utf-8") as f:
    content = f.read()

broken_line = "from langchain_community.chat_models.vertexai import ChatVertexAI"

if broken_line not in content:
    print("Nothing to patch — the broken import line was not found (already patched, or ragas layout changed).")
else:
    patched = content.replace(
        broken_line,
        (
            "try:\n"
            "    from langchain_community.chat_models.vertexai import ChatVertexAI\n"
            "except ImportError:\n"
            "    ChatVertexAI = None  # VertexAI not installed/available -- fine for Groq/OpenAI users\n"
        ),
    )
    with open(base_file, "w", encoding="utf-8") as f:
        f.write(patched)
    print("Patched successfully.")
    print("Now run: py evaluate_ragas.py")
