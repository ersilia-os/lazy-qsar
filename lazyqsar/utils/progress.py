from typing import Iterable, List, TypeVar, Optional
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
)

T = TypeVar("T") # lets do generics here for fun

def make_progress(transient: bool = True) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.fields[desc]}[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}" if "{task.total}" else "{task.completed}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=transient,
    )

def track_chunks(chunks: Iterable[T], desc: str, total: Optional[int] = None, transient: bool = True):
    with make_progress(transient) as progress:
        task = progress.add_task("work", total=total, desc=desc)
        for chunk in chunks:
            yield chunk
            try:
                n = len(chunk)
            except Exception:
                n = getattr(chunk, "shape", [1])[0] if hasattr(chunk, "shape") else 1
            progress.update(task, advance=int(n))
