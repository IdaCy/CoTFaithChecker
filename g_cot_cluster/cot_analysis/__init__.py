
"""Helper package for chain‑of‑thought faithfulness experiments"""
from .data_loading import load_segmented_jsons, build_category_sequences
from .metrics import (
    category_frequencies, transition_matrix, js_divergence,
    compute_entropy, lexical_yules_k, backtracking_correlation,
    xtp_train_test
)
from .visualization import (
    plot_category_bars, plot_transition_heatmap,
    plot_js_matrix, plot_length_entropy_scatter,
    plot_backtracking_accuracy, plot_xtp_roc
)
