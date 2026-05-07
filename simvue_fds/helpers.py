"""Helpers for extracting useful information from FDS files."""

import f90nml
import numpy
from fdsreader.slcf.slice import Slice


def create_heterogeneous_slice(
    slice: Slice,
) -> tuple[dict[str, numpy.ndarray], numpy.ndarray]:
    """Create a single array of values for a slice, with heterogeneous mesh sizes.

    As opposed to fdsreader's `to_global()` method, this function will only make the coarse
    mesh finer around areas which are required to maintain a constant sized 2D array. This
    results in a much less sparse array, saving memory.

    When making the mesh finer in a section, values are repeated to reconstruct the appearance
    of a coarser mesh.
    """
    coords: dict[str, numpy.ndarray] = slice.get_coordinates()
    dims = slice.extent_dirs
    values = numpy.zeros((len(slice.times), len(coords[dims[0]]), len(coords[dims[1]])))
    # Loop through subslices
    for subslice in slice.subslices:
        start_idx = []
        end_idx = []
        insert_indices = []

        # Get subslice data
        subslice_vals: numpy.ndarray = subslice.data

        # Loop through dimensions
        for i, dim in enumerate(dims):
            # Get coords for subslice
            sub_coords = subslice.get_coordinates()
            # Find indexes in global coords where subslice coords start and end
            start_idx.append(numpy.where(coords[dim] == sub_coords[dim][0])[0][0])
            end_idx.append(numpy.where(coords[dim] == sub_coords[dim][-1])[0][0])

            # Cut global coords to be same start and end as subslice coords
            trimmed_all_coords = coords[dim][start_idx[i] : end_idx[i] + 1]
            # Could use searchsorted to find indexes to insert elements into to maintain order of coords
            insert_indices.append(
                numpy.searchsorted(sub_coords[dim], trimmed_all_coords)
            )

        # Expand subslice values using the indices which maintain order
        subslice_expanded = subslice_vals[
            :, insert_indices[0][:, None], insert_indices[1][None, :]
        ]
        # Insert into correct place in grid
        values[
            :,
            start_idx[0] : start_idx[0] + subslice_expanded.shape[1],
            start_idx[1] : start_idx[1] + subslice_expanded.shape[2],
        ] = subslice_expanded

    return coords, values


def create_obst_mask(file_path: str, slice: Slice) -> numpy.ndarray:
    """Create a boolean mask of OBSTs for a given slice through the mesh.

    Note that OBST blocks which touch, but do not cross, the slice are not included.

    Parameters
    ----------
    file_path : str
        Path to the FDS input file.

    slice: Slice
        The slice to create a mask for, loaded by fdsreader

    Returns
    -------
    mask: numpy.ndarray
        A boolean mask representing areas covered by OBSTs


    """
    # Load coordinates from the slice
    all_coords = slice.get_coordinates()
    dims = slice.extent_dirs
    mask = numpy.full((len(all_coords[dims[0]]), len(all_coords[dims[1]])), False)

    # Find which dimension of the slice is a fixed value
    if "x" not in dims:
        fixed_val = all_coords["x"][0]
        fixed_idx = 0
    elif "y" not in dims:
        fixed_val = all_coords["y"][0]
        fixed_idx = 2
    else:
        fixed_val = all_coords["z"][0]
        fixed_idx = 4

    # Load FDS input file and loop through obstruction lines
    # Note that we do this instead of using `simulation.obstructions` from fdsreader,
    # as that seems to be inaccurate
    nml = f90nml.read(file_path)
    for key, val in nml.items():
        if key.lower() != "obst":
            continue
        obst_coords = list(val["xb"])
        # Check if obst exists over fixed dim
        if (
            obst_coords[fixed_idx] < fixed_val
            and obst_coords[fixed_idx + 1] > fixed_val
        ):
            # Remove fixed dimension coordinates from obst line
            obst_coords.pop(fixed_idx + 1)
            obst_coords.pop(fixed_idx)

            # Find indexes which correspond to the start and end points of the OBST within the slice
            # If the OBST does not exist inside the slice, it will return two equal values
            i_start = numpy.searchsorted(all_coords[dims[0]], obst_coords[0])
            i_end = numpy.searchsorted(
                all_coords[dims[0]], obst_coords[1], side="right"
            )

            j_start = numpy.searchsorted(all_coords[dims[1]], obst_coords[2])
            j_end = numpy.searchsorted(
                all_coords[dims[1]], obst_coords[3], side="right"
            )

            # Slice to replace OBSTs with True in the mask
            # If values above are equal nothing will happen,
            # hence this does not affect OBSTs which dont intersect the slice
            mask[
                i_start:i_end,
                j_start:j_end,
            ] = True

    return mask


def read_obst_rectangles(
    file_path: str,
    size: dict[str, float],
    heights: list[float],
) -> list[dict[str, tuple[float, float] | float]]:
    """Read all the obstacles from a specified horizontal region in an FDS file.

    This determines which areas are blocked from containing an item of a specified
    size. The inputs are the path, the size of the object, and the heights of the
    floors to investigate. This returns a list containing the positions and sizes
    of all rectangles that must be avoided, along with their base heights.

    Parameters
    ----------
    file_path : str
        Path to the FDS input file.

    size : dict[str, float]
        The 3-dimensional size/footprint of the object.

        Required keys are:
        - ``"x"`` : horizontal width in the x-direction
        - ``"y"`` : horizontal total width in the y-direction
        - ``"z"`` : vertical height in the z-direction

    heights : list[float]
        The floor levels to investigate.

    Returns
    -------
    list[dict[str, tuple[float, float] | float]]
        Rectangles representing regions in which the centre of the object cannot
        be located. Each item has the form::

            {
                "x": (x_min, x_max),
                "y": (y_min, y_max),
                "z": base_elevation,
            }

        A separate rectangle is returned for every matching obstacle and every
        queried base elevation, so the same obstacle may appear multiple times
        if it intersects multiple requested z layers.

    Raises
    ------
    RuntimeError
        Raised if any required size key (``"x"``, ``"y"``, or ``"z"``) is missing
        from the ``size`` dictionary.

    Notes
    -----
    - Only ``OBST`` entries in the input file are considered, and ``XB`` values
      are used to determine sizes.
    - Vertical intersection is tested using strict inequalities, so an obstacle
      whose bottom just touches the top of the object being placed is not included.
    - The returned footprints use the FDS ``XB`` x/y bounds, then pad them by half
      the supplied object size on each side.

    Examples
    --------
    Read obstacle rectangles intersecting two floors from an FDS file::

        rects = read_obst_rectangles(
            "case.fds",
            size={"x": 0.2, "y": 0.2, "z": 3.0},
            heights=[0.0, 3.0],
        )

    A returned entry has the form::

        {
            "x": (1.9, 4.1),
            "y": (0.4, 2.6),
            "z": 0.0,
        }

    """
    if "x" not in size:
        raise RuntimeError(
            '"x" width must be provided to read_obst_rectangles in the size dictionary'
        )
    if "y" not in size:
        raise RuntimeError(
            '"y" width must be provided to read_obst_rectangles in the size dictionary'
        )
    if "z" not in size:
        raise RuntimeError(
            '"z" height must be provided to read_obst_rectangles in the size dictionary'
        )

    x_radius = size["x"] / 2.0
    y_radius = size["y"] / 2.0

    nml = f90nml.read(file_path)

    rects: list[dict[str, tuple[float, float] | float]] = []

    for base in heights:
        top = base + size["z"]

        for key, val in nml.items():
            if key.lower() == "obst":
                coords = val["xb"]
                print(coords)

                if base < coords[5] and top > coords[4]:
                    rects.append(
                        {
                            "x": (coords[0] - x_radius, coords[1] + x_radius),
                            "y": (coords[2] - y_radius, coords[3] + y_radius),
                            "z": base,
                        }
                    )

    return rects


def read_floor_rectangles(
    file_path: str,
    heights: list[float],
) -> list[dict[str, tuple[float, float] | float]]:
    """Read all the floor rectangles from a specified set of heights in an FDS file.

    This determines which areas correspond exactly to floor surfaces at the
    specified levels. The inputs are the path and the heights of the floors to
    investigate. This returns a list containing the positions and sizes of all
    rectangles found at those floor heights.

    Parameters
    ----------
    file_path : str
        Path to the FDS input file.

    heights : list[float]
        The floor levels to investigate.

    Returns
    -------
    list[dict[str, tuple[float, float] | float]]
        Rectangles representing floor regions at the requested heights.
        Each item has the form::

            {
                "x": (x_min, x_max),
                "y": (y_min, y_max),
                "z": base_elevation,
            }

    Notes
    -----
    - Only ``OBST`` entries in the input file are considered, and ``XB`` values
      are used to determine sizes.
    - A rectangle is only returned when the top of the obstacle lies exactly at
      the requested floor height.
    - The returned footprints use the FDS ``XB`` x/y bounds directly, with no
      padding or offset applied.

    Examples
    --------
    Read floor rectangles from two levels in an FDS file::

        rects = read_floor_rectangles(
            "case.fds",
            heights=[0.0, 3.0],
        )

    A returned entry has the form::

        {
            "x": (0.0, 8.0),
            "y": (0.0, 6.0),
            "z": 3.0,
        }

    """
    nml = f90nml.read(file_path)

    rects: list[dict[str, tuple[float, float] | float]] = []

    for base in heights:
        for key, val in nml.items():
            if key.lower() == "obst":
                coords = val["xb"]

                if base == coords[5]:
                    rects.append(
                        {
                            "x": (coords[0], coords[1]),
                            "y": (coords[2], coords[3]),
                            "z": base,
                        }
                    )

    return rects
