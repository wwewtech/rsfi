"""
Runner for all unit tests in tests/
"""

import sys
from pathlib import Path

# Add src and root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_geometry import (
    test_normalization,
    test_geodesic_distance,
    test_log_map_axioms,
    test_exp_map,
)
from tests.test_whitening import (
    test_spherical_whitening_fit_transform,
    test_unfitted_transform_raises,
)
from tests.test_filter import (
    test_rsfi_pythagoras_decomposition,
    test_multidimensional_rsfi_filter,
)


def main():
    print("=" * 70)
    print("RUNNING RSFI UNIT TEST SUITE")
    print("=" * 70)

    print("\n[1/3] Testing Geometry (RiemannianSphere)...")
    test_normalization()
    test_geodesic_distance()
    test_log_map_axioms()
    test_exp_map()
    print("  [OK] Geometry tests passed successfully.")

    print("\n[2/3] Testing Whitening (SphericalWhitening)...")
    test_spherical_whitening_fit_transform()
    test_unfitted_transform_raises()
    print("  [OK] Whitening tests passed successfully.")

    print("\n[3/3] Testing Filters (RSFIFilter & MultiDimensionalRSFIFilter)...")
    test_rsfi_pythagoras_decomposition()
    test_multidimensional_rsfi_filter()
    print("  [OK] Filter tests passed successfully.")

    print("\n" + "=" * 70)
    print("ALL RSFI UNIT TESTS PASSED SUCCESSFULLY! (100%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
