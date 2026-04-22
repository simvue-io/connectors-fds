"""Helpers for extracting useful information from FDS files."""

import f90nml


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
