"""Immutable release bootstrap package.

Bootstrap modules are imported explicitly so loading a low-level contract never imports the
gateway and source-evolution composition graph.
"""

__all__: list[str] = []
