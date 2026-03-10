"""
rlaif/__init__.py
Package initializer — exposes top-level pipeline entry point.
"""
from .pipeline import run_pipeline, save_triplet

__all__ = ["run_pipeline", "save_triplet"]
