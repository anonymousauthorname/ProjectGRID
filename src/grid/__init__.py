"""GRID inference package.

The public GRIDOursMethod import is lazy so utility modules such as the Docker
entrypoint can run without eagerly initializing optional vLLM/Dropbox helpers.
"""

__all__ = ["GRIDOursMethod"]


def __getattr__(name):
    if name == "GRIDOursMethod":
        from src.grid.GRID_Ours import GRIDOursMethod

        return GRIDOursMethod
    raise AttributeError(name)
