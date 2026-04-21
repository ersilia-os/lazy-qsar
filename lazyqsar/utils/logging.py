import os
import sys
import logging
from typing import Optional

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.tree import Tree
from rich import box
from loguru import logger as _loguru

_loguru.remove()
_loguru.level("DEBUG", color="<cyan><bold>")
_loguru.level("INFO", color="<blue><bold>")
_loguru.level("WARNING", color="<white><bold><bg yellow>")
_loguru.level("ERROR", color="<white><bold><bg red>")
_loguru.level("CRITICAL", color="<white><bold><bg red>")
_loguru.level("SUCCESS", color="<black><bold><bg green>")

_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "{message}"
)


class Logger:
    def __init__(self):
        self.logger = _loguru
        self._console = Console(stderr=True, highlight=False)
        self._sink_id: Optional[int] = None
        self._verbose: bool = False

    @property
    def verbose(self) -> bool:
        return self._verbose

    def set_verbosity(self, verbose: bool):
        self._verbose = verbose
        if verbose and self._sink_id is None:
            self._sink_id = self.logger.add(
                sys.stderr,
                format=_FORMAT,
                colorize=True,
                level="DEBUG",
            )
        elif not verbose and self._sink_id is not None:
            try:
                self.logger.remove(self._sink_id)
            except Exception:
                pass
            self._sink_id = None

    def debug(self, text):
        self.logger.debug(text)

    def info(self, text):
        self.logger.info(text)

    def warning(self, text):
        self.logger.warning(text)

    def error(self, text):
        self.logger.error(text)

    def critical(self, text):
        self.logger.critical(text)

    def success(self, text):
        self.logger.success(text)

    def rule(self, title: str = "", style: str = "dim blue"):
        if not self._verbose:
            return
        if title:
            self._console.rule(f"[bold cyan]{title}[/]", style=style)
        else:
            self._console.rule(style=style)

    def profile_summary(self, profile) -> None:
        if not self._verbose:
            return

        task_label = {
            "classification": "Classification",
            "regression": "Regression",
        }.get(profile.task, profile.task)

        parts = [
            f"n={profile.n_samples:,}",
            f"p={profile.n_features:,}",
            f"n/p={profile.n_p_ratio:.1f}",
        ]
        if profile.task == "classification" and hasattr(profile, "imbalance_ratio"):
            parts.append(f"imbalance={profile.imbalance_ratio:.1f}:1")
        if profile.task == "regression" and hasattr(profile, "y_skewness"):
            parts.append(f"y_skewness={profile.y_skewness:.2f}")
        if profile.is_sparse_counts:
            parts.append("sparse_counts=True")
        if profile.binary_feature_fraction > 0.5:
            parts.append(f"binary_frac={profile.binary_feature_fraction:.2f}")

        sep = "  [dim]|[/]  "
        body = sep.join(f"[cyan]{p}[/]" for p in parts)
        self._console.print(f"[bold]{task_label}[/bold]  {body}")

    def dataset_table(self, X_shape, y=None, portfolio: list | None = None) -> None:
        """Display a compact dataset summary table."""
        if not self._verbose:
            return

        n_samples, n_features = X_shape
        table = Table(
            title="[bold]Dataset summary[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Metric", style="cyan", no_wrap=True, min_width=14)
        table.add_column("Value", justify="right", min_width=12)

        table.add_row("Samples", f"{n_samples:,}")
        table.add_row("Features", f"{n_features:,}")
        table.add_row("n/p ratio", f"{(n_samples / max(n_features, 1)):.2f}")

        if y is not None:
            y_arr = np.asarray(y)
            values, counts = np.unique(y_arr, return_counts=True)
            count_map = {int(v): int(c) for v, c in zip(values, counts)}
            pos = count_map.get(1, 0)
            neg = count_map.get(0, 0)
            table.add_row("Negatives", f"{neg:,}")
            table.add_row("Positives", f"{pos:,}")
            if len(y_arr) > 0:
                table.add_row("Positive rate", f"{(pos / len(y_arr)):.1%}")
            if neg > 0 and pos > 0:
                majority = max(pos, neg)
                minority = min(pos, neg)
                table.add_row("Class ratio", f"{majority / minority:.2f}:1")

        if portfolio is not None:
            table.add_row("Portfolio", ", ".join(portfolio))

        self._console.print(table)
        self._console.line()

    def batch_table(self, batches: list, strategy: str = "sequential") -> None:
        """Display a Rich table summarising batches.

        batches : list of dicts with keys 'n', 'n_pos', 'n_neg'.
                  For backward compat, a plain list of ints is also accepted.
        strategy : 'sequential' | 'imbalanced'
        """
        if not self._verbose:
            return

        # Normalise plain-int list to dict list
        if batches and isinstance(batches[0], int):
            batches = [{"n": n, "n_pos": None, "n_neg": None} for n in batches]

        strategy_label = {
            "sequential": "sequential",
            "imbalanced": "all-positives + neg partition",
        }.get(strategy, strategy)
        title = f"[bold]Training batches[/bold]  [dim]({strategy_label})[/dim]"

        table = Table(
            title=title,
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Batch", style="cyan", justify="right", width=7)
        table.add_column("n_total", justify="right", width=10)

        has_counts = batches[0]["n_pos"] is not None
        if has_counts:
            table.add_column("n_pos", justify="right", width=8)
            table.add_column("n_neg", justify="right", width=10)
            table.add_column("pos_rate", justify="right", width=10)

        for i, b in enumerate(batches):
            n = b["n"]
            row = [str(i), f"{n:,}"]
            if has_counts:
                n_pos = b["n_pos"]
                n_neg = b["n_neg"]
                rate = n_pos / n if n > 0 else 0.0
                row += [f"{n_pos:,}", f"{n_neg:,}", f"{rate:.2%}"]
            table.add_row(*row)

        self._console.print(table)
        self._console.line()

    def selector_table(
        self,
        portfolio: list,
        profile,
        scores: dict,
        reasons: list,
        selector_version: str,
    ) -> None:
        """Display a compact rule-based portfolio selection summary."""
        if not self._verbose:
            return

        table = Table(
            title=f"[bold]Portfolio selector[/bold]  [dim]({selector_version})[/dim]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Metric", style="cyan", no_wrap=True, min_width=16)
        table.add_column("Value", min_width=18)

        table.add_row("Portfolio", ", ".join(portfolio))
        table.add_row("Samples", f"{profile.n_samples:,}")
        table.add_row("Features", f"{profile.n_features:,}")
        table.add_row("n/p ratio", f"{profile.n_p_ratio:.2f}")
        table.add_row("Imbalance", f"{profile.imbalance_ratio:.2f}:1")
        table.add_row("Sparsity", f"{profile.sparsity:.3f}")
        table.add_row("Binary fraction", f"{profile.binary_feature_fraction:.3f}")
        table.add_row("Signal mean/p90", f"{profile.feature_signal_strength:.3f} / {profile.feature_signal_p90:.3f}")
        table.add_row(
            "Scores",
            f"lr={scores.get('lr', 0)}  xgb={scores.get('xgb', 0)}  rf={scores.get('rf', 0)}",
        )

        self._console.print(table)
        for reason in reasons:
            self._console.print(f"  [dim]- {reason}[/dim]")
        self._console.line()

    def portfolio_table(
        self,
        fast_scores: dict,
        params_map: dict,
        winner: str,
        threshold: float,
        default_score: float,
        n_tr: int,
        n_splits: int,
        skipped: list,
        model: str = "xgb",
    ) -> None:
        if not self._verbose:
            return

        is_rf = (model == "rf")
        title = (
            "[bold]Portfolio — OOB comparison[/bold]"
            if is_rf else
            "[bold]Portfolio — Stage 1 comparison[/bold]"
        )
        table = Table(
            title=title,
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Preset", style="cyan", no_wrap=True, min_width=9)
        if is_rf:
            table.add_column("n_est", justify="right", width=6, no_wrap=True)
            table.add_column("mln",   justify="right", width=6, no_wrap=True)
            table.add_column("max_f", justify="right", width=7, no_wrap=True)
        else:
            table.add_column("LR",    justify="right", width=7, no_wrap=True)
            table.add_column("Depth", justify="right", width=6, no_wrap=True)
        table.add_column("Score",  justify="right", width=8)
        table.add_column("Gain",   justify="right", width=8)
        table.add_column("Status", no_wrap=True, min_width=14)

        preset_order = ["heuristic", "default", "flaml", "autogluon"]

        for name in preset_order:
            score = fast_scores.get(name, float("nan"))
            params = params_map.get(name, {})
            is_nan = score != score

            if is_rf:
                col_a = str(params.get("n_estimators", "?"))
                col_b = str(params.get("max_leaf_nodes", "—"))
                col_c = str(params.get("max_features", "?"))
                param_cols = [col_a, col_b, col_c]
            else:
                lr = params.get("learning_rate", float("nan"))
                col_a = f"{lr:.4f}" if lr == lr else "—"
                if params.get("grow_policy") == "lossguide":
                    col_b = f"{params.get('max_leaves', '?')}L"
                else:
                    col_b = str(params.get("max_depth", "?"))
                param_cols = [col_a, col_b]

            score_str = f"{score:.4f}" if not is_nan else "—"

            if name == "default" or is_nan:
                gain_str = "—"
                gain_style = "dim"
                gain = 0.0
            else:
                gain = score - default_score
                gain_str = f"{gain:+.4f}"
                gain_style = "green" if gain > 0 else "red"

            if is_nan:
                status = "[dim]skipped[/dim]"
            elif name == winner:
                status = "[bold green]✓ selected[/]" if name != "default" else "[bold green]✓ default[/]"
            elif name == "default":
                status = "[dim]baseline[/dim]"
            else:
                if gain > 0 and gain < threshold:
                    status = "[yellow]< threshold[/yellow]"
                elif gain <= 0:
                    status = "[dim]worse[/dim]"
                else:
                    status = "[dim]—[/dim]"

            row_style = "bold" if name == winner else ""
            table.add_row(name, *param_cols, score_str, f"[{gain_style}]{gain_str}[/]", status, style=row_style)

        self._console.print(table)
        footer_parts = [f"threshold = {threshold:.4f}", f"n_train = {n_tr:,}"]
        if not is_rf:
            footer_parts.append(f"{n_splits} split(s) averaged")
        if is_rf:
            footer_parts.append("mln = max_leaf_nodes")
        self._console.print("  [dim]" + "  |  ".join(footer_parts) + "[/dim]")
        self._console.line()


    def inner_pooler_table(
        self,
        portfolio: list,
        n_samples: int,
        oof_aucs: list | None = None,
        meta_auc: float | None = None,
        mean_weights: list | None = None,
        std_weights: list | None = None,
    ) -> None:
        """Display a Rich table summarising the ensemble pooler."""
        if not self._verbose:
            return

        gating = mean_weights is not None
        title = (
            "[bold]Ensemble heads[/bold] [dim](per-sample gating)[/dim]"
            if gating else
            "[bold]Ensemble heads[/bold]"
        )

        table = Table(
            title=title,
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Head", style="cyan", no_wrap=True, min_width=6)
        if oof_aucs is not None:
            table.add_column("OOF score", justify="right", min_width=10)
        if mean_weights is not None:
            table.add_column("Mean weight", justify="right", min_width=12)
        if std_weights is not None:
            table.add_column("Weight std", justify="right", min_width=11)

        for i, head in enumerate(portfolio):
            row = [head]
            if oof_aucs is not None:
                row.append(f"{oof_aucs[i]:.4f}")
            if mean_weights is not None:
                row.append(f"{mean_weights[i]:.4f}")
            if std_weights is not None:
                row.append(f"{std_weights[i]:.4f}")
            table.add_row(*row)

        self._console.print(table)

        footer_parts = [f"n = {n_samples:,}"]
        if meta_auc is not None:
            footer_parts.append(f"meta AUC = {meta_auc:.4f}")
        self._console.print("  [dim]" + "  |  ".join(footer_parts) + "[/dim]")
        self._console.line()

    def descriptor_table(self, rows: list) -> None:
        """
        Compact summary of all descriptors used in a LazyClassifierQSAR fit.

        rows : list of dicts with keys:
            name, n_features, sparsity, feat_time,
            ad_n_components, ad_cal_min, ad_cal_max
        """
        if not self._verbose:
            return

        table = Table(
            title="[bold]Descriptor summary[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        has_oof = any("oof_auc" in r for r in rows)
        has_train = any("train_auc" in r for r in rows)
        has_quality = any("quality_auc" in r for r in rows)
        has_active = any("active" in r for r in rows)

        table.add_column("Descriptor",  style="cyan", no_wrap=True, min_width=12)
        table.add_column("p",           justify="right", min_width=6)
        table.add_column("sparsity",    justify="right", min_width=8)
        table.add_column("feat time",   justify="right", min_width=9)
        table.add_column("AD comps",    justify="right", min_width=9)
        table.add_column("AD cal max",  justify="right", min_width=10)
        if has_oof:
            table.add_column("OOF AUC",     justify="right", min_width=9)
        if has_train:
            table.add_column("Train AUC",   justify="right", min_width=10)
        if has_quality:
            table.add_column("Quality AUC", justify="right", min_width=12)
        if has_active:
            table.add_column("Active",      justify="center", min_width=7)

        for r in rows:
            row = [
                r["name"],
                f"{r['n_features']:,}",
                f"{r['sparsity']:.3f}",
                f"{r['feat_time']:.1f}s",
                str(r["ad_n_components"]),
                f"{r['ad_cal_max']:.3f}",
            ]
            if has_oof:
                row.append(f"{r['oof_auc']:.4f}")
            if has_train:
                row.append(f"{r['train_auc']:.4f}")
            if has_quality:
                row.append(f"{r['quality_auc']:.4f}")
            if has_active:
                row.append("[bold green]✓[/]" if r["active"] else "[dim]✗[/]")
            table.add_row(*row)

        self._console.print(table)
        self._console.line()

    def ad_weights_table(self, rows: list, n_samples: int) -> None:
        """
        rows : list of dicts with keys:
            name, ad_mean, ad_std, ad_min, ad_max,
            weight_mean, weight_std, wins, pred_mean
        """
        if not self._verbose:
            return

        table = Table(
            title="[bold]Ensemble predictions[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        has_oof = any("oof_auc" in r for r in rows)

        table.add_column("Descriptor",       style="cyan", no_wrap=True, min_width=12)
        if has_oof:
            table.add_column("OOF AUC",      justify="right", min_width=9)
        table.add_column("AD mean±std",      justify="right", min_width=13)
        table.add_column("AD [min, max]",    justify="right", min_width=14)
        table.add_column("Final W mean±std", justify="right", min_width=16)
        table.add_column("Wins",             justify="right", min_width=8)
        table.add_column("P(y=1) mean",      justify="right", min_width=12)

        for r in rows:
            wins_pct = 100.0 * r["wins"] / n_samples if n_samples > 0 else 0.0
            row = [r["name"]]
            if has_oof:
                row.append(f"{r['oof_auc']:.4f}")
            row += [
                f"{r['ad_mean']:.3f}±{r['ad_std']:.3f}",
                f"[{r['ad_min']:.3f}, {r['ad_max']:.3f}]",
                f"{r['weight_mean']:.3f}±{r['weight_std']:.3f}",
                f"{r['wins']} ({wins_pct:.0f}%)",
                f"{r['pred_mean']:.3f}",
            ]
            table.add_row(*row)

        self._console.print(table)
        self._console.print(f"  [dim]n={n_samples:,}[/dim]")
        self._console.line()

    def timing_table(self, steps: list) -> None:
        """
        Print a per-step timing breakdown table.

        steps : list of (name, seconds, is_subtask) tuples.
                is_subtask=True → indented row, % column left blank.
        Only rendered when verbose=True.
        """
        if not self._verbose:
            return
        top_level = [t for _, t, sub in steps if not sub]
        total = sum(top_level) if top_level else 0.0
        table = Table(
            title="[bold]Fit timing breakdown[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Step", min_width=38)
        table.add_column("Time (s)", justify="right", min_width=9)
        table.add_column("%", justify="right", min_width=5)
        for name, t, is_subtask in steps:
            label = f"  {name}" if is_subtask else name
            t_str = f"{t:.2f}"
            pct_str = "" if is_subtask else (f"{100*t/total:.0f}%" if total > 0 else "—")
            style = "dim" if is_subtask else ""
            table.add_row(label, t_str, pct_str, style=style)
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total:.2f}[/bold]",
            "[bold]100%[/bold]",
        )
        self._console.print(table)
        self._console.line()

    def dir_tree(self, directory: str) -> None:
        """Print a rich directory tree with file sizes for the saved model folder."""
        root_label = f"[bold]{os.path.basename(directory)}[/bold]/"
        tree = Tree(root_label, guide_style="dim")

        def _human_size(num_bytes: int) -> str:
            size = float(num_bytes)
            if size < 1024:
                return f"{int(size)} B"
            if size < 1024**2:
                return f"{size / 1024:.1f} KB"
            if size < 1024**3:
                return f"{size / 1024**2:.2f} MB"
            return f"{size / 1024**3:.2f} GB"

        def _add(node, path):
            entries = sorted(os.listdir(path))
            subdirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
            files   = [e for e in entries if os.path.isfile(os.path.join(path, e))]
            for d in subdirs:
                child = node.add(f"[bold cyan]{d}/[/bold cyan]")
                _add(child, os.path.join(path, d))
            for f in files:
                size = os.path.getsize(os.path.join(path, f))
                node.add(f"[dim]{f}[/dim]  [green]({_human_size(size)})[/green]")

        _add(tree, directory)
        self._console.print(tree)


logger = Logger()
console = Console(stderr=True, highlight=False)
