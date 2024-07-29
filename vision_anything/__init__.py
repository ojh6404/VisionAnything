import os

CACHE_DIR = os.path.expanduser("~/.cache/vision_anything")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

CHECKPOINT_ROOT = os.path.join(CACHE_DIR, "checkpoints")
if not os.path.exists(CHECKPOINT_ROOT):
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
THIRD_PARTY_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, "..", "third_party"))
