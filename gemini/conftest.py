"""Root conftest.py — ensures the gemini project's src/ package takes precedence
over any same-named src/ package registered via editable installs in sys.meta_path.

This runs before test collection, so we can clear any cached src.* modules and
insert our own MetaPathFinder first."""
import sys
import pathlib
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_file_location

_GEMINI_ROOT = pathlib.Path(__file__).parent.resolve()
_SRC_ROOT = _GEMINI_ROOT / "src"


class _GeminiSrcFinder:
    """MetaPathFinder that resolves `src` and `src.*` to this project's src/."""

    @classmethod
    def find_spec(cls, fullname: str, path=None, target=None):
        if fullname == "src":
            init = _SRC_ROOT / "__init__.py"
            if init.exists():
                return spec_from_file_location(fullname, init,
                                               submodule_search_locations=[str(_SRC_ROOT)])
        if fullname.startswith("src."):
            child = fullname[4:]  # strip "src."
            # Only handle direct children (no further dots)
            if "." not in child:
                candidate = _SRC_ROOT / f"{child}.py"
                if candidate.exists():
                    return spec_from_file_location(fullname, candidate)
        return None


# Remove any cached src.* modules from a previous (wrong) import
for _key in list(sys.modules.keys()):
    if _key == "src" or _key.startswith("src."):
        del sys.modules[_key]

# Insert before any editable-install finders so our src/ wins.
sys.meta_path.insert(0, _GeminiSrcFinder)
