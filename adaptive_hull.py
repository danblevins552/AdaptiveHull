import numpy as np
import sympy as sp
from itertools import product
from lower_hull import lower_convex_hull

def compute_points(space, labels, phase_eq_dict):
    """
    Compute the energy for each point in `space` given phase-specific symbolic expressions.

    This function preserves the order of `space` by storing the results back into the pre-allocated
    `points` array according to each point's index.

    Parameters
    ----------
    space : np.ndarray, shape (n, m)
        The array of points for which energies should be computed. Each row is one point.
    labels : np.ndarray, shape (n,)
        Phase labels corresponding to each point in `space`. The label determines which symbolic
        equation to use for energy calculation.
    phase_eq_dict : dict
        A dictionary mapping from phase label (string) to a sympy expression (e.g. sp.Symbol('x0') + 2*sp.Symbol('x1')).

    Returns
    -------
    points : np.ndarray, shape (n, m+1)
        The first m columns are the original coordinates of `space`, and the last column is
        the computed energy. Rows appear in the same order as `space`.
    new_labels : np.ndarray, shape (n,)
        The phase labels in the same order as `space`.
    """
    n, m = space.shape

    # Preallocate the points array: each row has m space coordinates plus 1 for the energy value.
    points = np.empty((n, m + 1), dtype=space.dtype)
    # Preallocate an output label array with the same shape as the original labels.
    new_labels = np.empty(n, dtype=labels.dtype)

    # Loop over each phase in the dictionary
    for phase, func in phase_eq_dict.items():
        # Find indices corresponding to this phase
        phase_idx = np.where(labels == phase)[0]
        if phase_idx.size == 0:
            continue

        # Subset of `space` corresponding to the current phase
        subspace = space[phase_idx]

        # Symbolically compute energies for these points
        energies = func(subspace)
        
        # Ensure energies is a 1D array
        energies = energies.ravel()

        # Store the (x-coordinates, energy) back in the correct positions
        points[phase_idx, :m] = subspace
        points[phase_idx, m] = energies

        # Also assign the label
        new_labels[phase_idx] = phase

    return points, new_labels


def find_multiphase_indices(lower_hull_simplices, labels):
    """
    Determine which simplices on the lower hull contain more than one phase.

    Parameters
    ----------
    lower_hull_simplices : np.ndarray, shape (n_simplices, dim+1)
        Array of vertex indices forming the simplices on the lower hull.
    labels : np.ndarray, shape (n_points,)
        Phase labels for each point in the dataset.

    Returns
    -------
    multiphase_indices : np.ndarray
        The sorted, unique indices of all points (vertices) that lie on a simplex
        containing more than one phase label.
    """
    # Labels of all vertices for each simplex
    simplex_labels = labels[lower_hull_simplices]

    # Identify simplices whose vertices do NOT all share the same label
    multiphase_simplices = lower_hull_simplices[
        ~np.all(simplex_labels == simplex_labels[:, 0][:, None], axis=1)
    ]

    # Flatten the vertex indices and remove duplicates
    multiphase_indices = np.unique(multiphase_simplices.ravel())
    return multiphase_indices


def unique_preserve_order(a):
    """
    Return the unique rows of a while preserving the order of their first occurrence.

    This leverages a 'void' view of each row so that rows can be treated as single elements
    for deduplication, but ensures that the first time a row appears is the version retained.

    Parameters
    ----------
    a : np.ndarray, shape (n, m)
        Input 2D array whose unique rows should be extracted.

    Returns
    -------
    unique_rows : np.ndarray, shape (k, m)
        The unique rows in `a`, preserving the order of first appearance.
    indices : np.ndarray, shape (k,)
        The indices of the first occurrence for each unique row in `a`.
    """
    # Convert each row to a single void element so np.unique works row-wise
    a_view = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    )
    _, idx = np.unique(a_view, return_index=True)
    # Sort these indices so we retain the first occurrence in the original order
    order = np.sort(idx)
    return a[order], order


def expand_space(space, labels, alpha, bound_lower, bound_upper):
    """
    Expand the input space by adding all combinations of offsets (0, +alpha, -alpha)
    to each dimension of each point, then deduplicate while preserving the first occurrence.

    Parameters
    ----------
    space : np.ndarray, shape (n, m)
        Original points to be expanded.
    labels : np.ndarray, shape (n,)
        Phase labels corresponding to each point in `space`.
    alpha : float
        Offset magnitude to add or subtract. For each dimension, we use {0, +alpha, -alpha}.
    bound_lower : float or np.ndarray
        Lower bound(s) for clipping. Can be a scalar or 1D array of length m.
    bound_upper : float or np.ndarray
        Upper bound(s) for clipping. Can be a scalar or 1D array of length m.

    Returns
    -------
    unique_expanded_space : np.ndarray, shape (k, m)
        Deduplicated array of expanded points, clipped to the given bounds.
    unique_labels : np.ndarray, shape (k,)
        Labels corresponding to each row in `unique_expanded_space`.
    zero_offset_indices : np.ndarray
        Indices in `unique_expanded_space` that came from the 0 offset (i.e. unchanged from `space`).
    """
    n, m = space.shape

    # Generate all offset combinations of 0, +alpha, and -alpha for each of m dimensions
    offsets = np.array(list(product([0, alpha, -alpha], repeat=m)), dtype=space.dtype)
    k = offsets.shape[0]  # 3**m combinations

    # Apply each offset combination to each row in space
    broadcasted = space[:, None, :] + offsets  # shape: (n, k, m)
    expanded_space = broadcasted.reshape(-1, m)

    # Clip each coordinate to remain within [bound_lower, bound_upper]
    expanded_space = np.clip(expanded_space, bound_lower, bound_upper)

    # Replicate labels and offsets for the expanded points
    labels_expanded = np.repeat(labels, k)
    offsets_expanded = np.tile(offsets, (n, 1))

    # Remove duplicate rows, preserving the order of their first appearance
    unique_expanded_space, unique_idx = unique_preserve_order(expanded_space)
    unique_labels = labels_expanded[unique_idx]
    unique_offsets_used = offsets_expanded[unique_idx]

    # Identify which expanded rows came from a zero offset (i.e. no change)
    zero_offset_mask = np.all(unique_offsets_used == 0, axis=1)
    zero_offset_indices = np.where(zero_offset_mask)[0]

    return unique_expanded_space, unique_labels, zero_offset_indices


def adaptive_hull(sparse_space, phase_eq_dict, bound_lower, bound_upper,
                  alpha=0.1, refinement_threshold=0):
    """
    Repeatedly refine the set of points suspected to lie on the multiphase boundary
    by expanding their vicinity and checking which points remain on the boundary.

    1. Compute energies for each phase at every point in `sparse_space`.
    2. Build a lower convex hull of all points; find which points share multiple phases on that hull.
    3. Expand those multiphase boundary points by +/- alpha in each dimension.
    4. Recompute energies and repeat until the fraction of old boundary points that remain
       is below `refinement_threshold`.

    Parameters
    ----------
    sparse_space : np.ndarray, shape (n, m)
        Initial set of points in the space.
    phase_eq_dict : dict
        A dictionary mapping phase labels to sympy expressions describing their energy.
    bound_lower : float or np.ndarray
        Lower bound(s) for the expansions, used in np.clip.
    bound_upper : float or np.ndarray
        Upper bound(s) for the expansions, used in np.clip.
    alpha : float, default=0.1
        Expansion offset used in expand_space. Each dimension is offset by {0, +alpha, -alpha}.
    refinement_threshold : float, default=0
        If the fraction of zero-offset points still on the boundary is greater than 0,
        the algorithm continues refining. If you want to stop earlier, set a higher threshold (e.g. 0.2).

    Returns
    -------
    points, labels : np.ndarray, np.ndarray
        The final set of points on the multiphase boundary and their corresponding phase labels.
    """
    n, m = sparse_space.shape
    unique_labels = np.array(list(phase_eq_dict.keys()))

    # Assign a label to each point for each phase (so we compute each phase's energy).
    labels = np.tile(unique_labels, n)
    # Repeat each point in sparse_space once for each phase
    sparse_space = np.repeat(sparse_space, len(unique_labels), axis=0)
    points, labels = compute_points(sparse_space, labels, phase_eq_dict)

    # Compute the lower convex hull and find which points are multiphase
    lower_hull_simplices = lower_convex_hull(points, True)
    multiphase_indices = find_multiphase_indices(lower_hull_simplices, labels)

    # Restrict to just the multiphase points (columns: all but the last are coords, last is energy)
    space = points[:, :-1][multiphase_indices]
    labels = labels[multiphase_indices]

    refinement = 1
    while refinement > refinement_threshold:
        n, m = space.shape  # May have changed size in the previous iteration

        # Expand the space around the suspected multiphase boundary
        expanded_space, expanded_labels, zero_offset_indices = expand_space(
            space, labels, alpha, bound_lower, bound_upper
        )

        # Compute energies for the newly expanded space
        points, labels = compute_points(expanded_space, expanded_labels, phase_eq_dict)
        
        # Build the hull again and see which points are on a multiphase simplex
        lower_hull_simplices = lower_convex_hull(points, True)
        multiphase_indices = find_multiphase_indices(lower_hull_simplices, labels)

        # Fraction of old boundary points (zero_offset_indices) that remain on the multiphase boundary
        # The bigger the intersection, the more points remain. We measure "refinement" as how many have left.
        old_on_boundary = len(np.intersect1d(zero_offset_indices, multiphase_indices))
        refinement = 1 - old_on_boundary / len(multiphase_indices)

        # Update space and labels to focus on the new boundary
        space = points[:, :-1][multiphase_indices]
        labels = labels[multiphase_indices]

    return points[multiphase_indices], labels