def search_cv_splits(n_samples: int) -> int:
    """Return n_splits for StratifiedShuffleSplit (test_size=0.2) used in find_params."""
    if n_samples < 2000:
        return 3
    else:
        return 2
