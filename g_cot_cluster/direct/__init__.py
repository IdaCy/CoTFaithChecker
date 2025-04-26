"""CoT faithfulness analysis – convenience re‑exports."""
from importlib import util, machinery
from types import ModuleType
from pathlib import Path
import sys

# --- dynamic sub‑module loader so users can just copy this one
#     canvas into a folder and still `import cot_analysis`.
_pkg_dir = Path(__file__).parent
for _file in _pkg_dir.glob("*.py"):
    if _file.name == "__init__.py":
        continue
    name = f"{__name__}.{_file.stem}"
    spec = util.spec_from_file_location(name, _file)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore

del util, machinery, ModuleType, Path, sys, _pkg_dir, _file, spec, module, name

# Re‑export the public API
from .data_utils import load_data, merge_accuracy
from .metrics import add_basic_metrics
from .stats import (categorical_chi2, permutation_test,
                    global_transition_matrix, plot_transition_heatmap)
from .models import train_category_classifier, evaluate_classifier

__all__ = [
    "load_data",
    "merge_accuracy",
    "add_basic_metrics",
    "categorical_chi2",
    "permutation_test",
    "global_transition_matrix",
    "plot_transition_heatmap",
    "train_category_classifier",
    "evaluate_classifier",
]