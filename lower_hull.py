import numpy as np
from scipy.spatial import ConvexHull

def lower_convex_hull(points, check_collinear=True):
    """
    Compute the lower convex hull of a set of points in N-dimensional space.
    Assumes the last coordinate of each point is the 'energy' dimension.

    Parameters
    ----------
    points : (M, N) ndarray
        Array of M points in N-dimensional space. The last dimension (N-th) 
        is treated as the 'energy' coordinate.

    Returns
    -------
    lower_hull : (K, D) ndarray of int
        Array of indices (simplices) describing the points that form the 
        lower convex hull in N-dimensional space. Each row of 'lower_hull'
        is a facet (simplex) with D indices into 'points'.
    """

    projected_points = points[:, :-1]  # everything except the last dimension
    ref_point = projected_points[0]
    rank = None
    # Check if the points (minus the last dimension) are collinear
    if check_collinear:
        rank = np.linalg.matrix_rank(projected_points - ref_point)

    # If collinear, use the end points as the boundary
    if rank == 1:
        # Sort all points by distance from the first point, take first and last
        dists = np.linalg.norm(projected_points - ref_point, axis=1)
        idx_sorted = np.argsort(dists)
        boundary_indices = np.array([idx_sorted[0], idx_sorted[-1]])
    # If not colinear, compute the convex hull of the projection
    else:
        boundary_hull = ConvexHull(projected_points)
        boundary_indices = np.unique(boundary_hull.simplices)

    # Create "upper" points only for those boundary indices
    fake_points = points[boundary_indices].copy()
    fake_points[:, -1] += 5e5  # shift energies upward
    combined_points = np.concatenate([points, fake_points], axis=0)

    # Compute the hull of this combined set
    hull = ConvexHull(combined_points)

    # Identify facets whose vertices are all in the original set
    mask = np.all(hull.simplices < len(points), axis=1)
    lower_hull = hull.simplices[mask]

    return lower_hull