"""Where report outputs go: figures, tables and the data behind them, under the paper directory."""
import os


def paper_file(out, name):
    """out/figures/fig_*, out/tables/tab_*, out/data/<anything else>; creates the subdirectory."""
    sub = "figures" if name.startswith("fig_") else "tables" if name.startswith("tab_") else "data"
    os.makedirs(os.path.join(out, sub), exist_ok=True)
    return os.path.join(out, sub, name)
