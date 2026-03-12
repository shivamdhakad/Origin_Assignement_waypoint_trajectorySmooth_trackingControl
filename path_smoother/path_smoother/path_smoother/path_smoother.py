"""
Path Smoother Module
====================
Implements cubic spline interpolation to smooth a list of 2D waypoints
into a continuous, differentiable path.

Algorithm: Scipy CubicSpline with arc-length parameterization.
  - Parameterize waypoints by cumulative chord length (arc-length approx)
  - Fit independent splines for x(t) and y(t)
  - Sample densely to produce a smooth path
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Tuple


def compute_arc_length_parameterization(waypoints: List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute cumulative chord-length distances as the spline parameter.

    Arc-length parameterization prevents uneven spacing artifacts that
    appear when using simple index-based parameterization.

    Args:
        waypoints: List of (x, y) tuples.

    Returns:
        1D array of cumulative distances, starting at 0.
    """
    pts = np.array(waypoints)
    diffs = np.diff(pts, axis=0)                    # vectors between consecutive points
    distances = np.linalg.norm(diffs, axis=1)       # Euclidean distances
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    return cumulative


def smooth_path(
    waypoints: List[Tuple[float, float]],
    num_samples: int = 500,
) -> List[Tuple[float, float]]:
    """
    Smooth a list of 2D waypoints using cubic spline interpolation.

    Args:
        waypoints:   List of (x, y) tuples representing the coarse path.
        num_samples: Number of points to sample along the smooth spline.

    Returns:
        List of (x, y) tuples representing the smoothed path.

    Raises:
        ValueError: If fewer than 2 waypoints are provided.
    """
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints are required for smoothing.")

    if len(waypoints) == 2:
        # Trivial case: straight line between two points
        xs = np.linspace(waypoints[0][0], waypoints[1][0], num_samples)
        ys = np.linspace(waypoints[0][1], waypoints[1][1], num_samples)
        return list(zip(xs.tolist(), ys.tolist()))

    t = compute_arc_length_parameterization(waypoints)
    pts = np.array(waypoints)

    # Fit a cubic spline for each axis independently
    cs_x = CubicSpline(t, pts[:, 0])
    cs_y = CubicSpline(t, pts[:, 1])

    # Sample uniformly along the arc-length parameter
    t_fine = np.linspace(t[0], t[-1], num_samples)
    x_smooth = cs_x(t_fine)
    y_smooth = cs_y(t_fine)

    return list(zip(x_smooth.tolist(), y_smooth.tolist()))


def compute_path_length(path: List[Tuple[float, float]]) -> float:
    """
    Compute the total arc length of a path.

    Args:
        path: List of (x, y) tuples.

    Returns:
        Total Euclidean path length.
    """
    pts = np.array(path)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))
