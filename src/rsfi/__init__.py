"""
RSFI: Riemannian System Fidelity Index package.
"""

from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening
from rsfi.filter import RSFIFilter, MultiDimensionalRSFIFilter
from rsfi.engine import (
    ProductionSFIEngine,
    SFIBenchmarkRunner,
    SentenceTelemetry,
    BenchmarkSummary,
)

__version__ = "0.1.0"
__all__ = [
    "RiemannianSphere",
    "SphericalWhitening",
    "RSFIFilter",
    "MultiDimensionalRSFIFilter",
    "ProductionSFIEngine",
    "SFIBenchmarkRunner",
    "SentenceTelemetry",
    "BenchmarkSummary",
]
