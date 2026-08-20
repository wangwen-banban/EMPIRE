from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent.parent / "empire" / "models" / "depth_anything_3"

# Let imports like `depth_anything_3.cfg` resolve to the vendored DA3 package.
__path__ = [str(_pkg_dir)]
__file__ = str(_pkg_dir / "__init__.py")
