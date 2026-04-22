import textwrap

import pytest

from simvue_fds import helpers


def _write_fds(tmp_path, text: str):
    path = tmp_path / "case.fds"
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("size", "message"),
    [
        ({"y": 1.0, "z": 2.0}, '"x" width'),
        ({"x": 1.0, "z": 2.0}, '"y" width'),
        ({"x": 1.0, "y": 2.0}, '"z" height'),
    ],
)
def test_read_obst_rectangles_raises_error_if_size_key_missing(size, message):
    with pytest.raises(
        RuntimeError,
        match=f"{message} must be provided to read_obst_rectangles in the size dictionary",
    ):
        helpers.read_obst_rectangles("dummy.fds", size=size, heights=[0.0])


def test_read_obst_rectangles_reads_matching_obstacles_and_applies_padding(tmp_path):
    path = _write_fds(
        tmp_path,
        """
        &OBST XB=1.0,3.0,10.0,14.0,0.5,2.0 /
        &OBST XB=5.0,6.0,7.0,8.0,4.0,5.0 /
        """,
    )

    result = helpers.read_obst_rectangles(
        str(path),
        size={"x": 2.0, "y": 4.0, "z": 1.0},
        heights=[1.0],
    )

    assert result == [{"x": (0.0, 4.0), "y": (8.0, 16.0), "z": 1.0}]


@pytest.mark.parametrize(
    ("size_z", "obst_base", "obst_top", "expected_count"),
    [
        (1.0, 0.0, 3.0, 1),  # object fully inside obstacle
        (3.0, 2.0, 3.0, 1),  # obstacle fully inside object
        (1.0, 0.0, 1.5, 1),  # partial overlap from below
        (1.0, 1.5, 3.0, 1),  # partial overlap from above
        (1.0, 2.0, 3.0, 0),  # obstacle bottom touches object top
        (1.0, 0.0, 1.0, 0),  # obstacle top touches object base
        (1.0, 2.1, 3.0, 0),  # fully above
        (1.0, -2.0, 0.9, 0),  # fully below
    ],
)
def test_read_obst_rectangles_checks_strict_vertical_intersection(
    tmp_path, size_z, obst_base, obst_top, expected_count
):
    path = _write_fds(
        tmp_path,
        f"""
        &OBST XB=10.0,20.0,30.0,40.0,{obst_base},{obst_top} /
        """,
    )

    result = helpers.read_obst_rectangles(
        str(path),
        size={"x": 2.0, "y": 2.0, "z": size_z},
        heights=[1.0],
    )

    assert len(result) == expected_count
    if expected_count:
        assert result == [{"x": (9.0, 21.0), "y": (29.0, 41.0), "z": 1.0}]


def test_read_obst_rectangles_returns_both_matching_obstacles_at_one_floor(tmp_path):
    path = _write_fds(
        tmp_path,
        """
        &OBST XB=0.0,1.0,0.0,1.0,0.0,2.0 /
        &OBST XB=2.0,4.0,3.0,5.0,0.5,3.0 /
        &OBST XB=10.0,11.0,10.0,11.0,3.0,4.0 /
        """,
    )

    result = helpers.read_obst_rectangles(
        str(path),
        size={"x": 2.0, "y": 2.0, "z": 1.0},
        heights=[1.0],
    )

    assert result == [
        {"x": (-1.0, 2.0), "y": (-1.0, 2.0), "z": 1.0},
        {"x": (1.0, 5.0), "y": (2.0, 6.0), "z": 1.0},
    ]


def test_read_obst_rectangles_returns_one_result_per_matching_height(tmp_path):
    path = _write_fds(
        tmp_path,
        """
        &OBST XB=0.0,1.0,0.0,1.0,0.0,5.0 /
        """,
    )

    result = helpers.read_obst_rectangles(
        str(path),
        size={"x": 0.0, "y": 0.0, "z": 1.0},
        heights=[0.0, 2.0, 4.5],
    )

    assert result == [
        {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": 0.0},
        {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": 2.0},
        {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": 4.5},
    ]


def test_read_obst_rectangles_with_empty_heights_returns_empty_list(tmp_path):
    path = _write_fds(
        tmp_path,
        """
        &OBST XB=0.0,1.0,0.0,1.0,0.0,2.0 /
        """,
    )

    result = helpers.read_obst_rectangles(
        str(path),
        size={"x": 1.0, "y": 1.0, "z": 1.0},
        heights=[],
    )

    assert result == []


def test_read_floor_rectangles_returns_exact_matches_at_requested_heights(tmp_path):
    path = _write_fds(
        tmp_path,
        """
        &OBST XB=0.0,8.0,0.0,6.0,0.0,3.0 /
        &OBST XB=2.0,3.0,7.0,9.0,2.0,3.0 /
        &OBST XB=1.0,2.0,1.0,2.0,4.0,5.0 /
        """,
    )

    result = helpers.read_floor_rectangles(str(path), heights=[3.0, 5.0, 7.0])

    assert result == [
        {"x": (0.0, 8.0), "y": (0.0, 6.0), "z": 3.0},
        {"x": (2.0, 3.0), "y": (7.0, 9.0), "z": 3.0},
        {"x": (1.0, 2.0), "y": (1.0, 2.0), "z": 5.0},
    ]


@pytest.mark.parametrize(
    ("top", "matches"), [(3.0, True), (3.0000000001, False), (2.9999999999, False)]
)
def test_read_floor_rectangles_uses_exact_equality(tmp_path, top, matches):
    path = _write_fds(
        tmp_path,
        f"""
        &OBST XB=1.0,2.0,3.0,4.0,0.0,{top} /
        """,
    )

    result = helpers.read_floor_rectangles(str(path), heights=[3.0])

    if matches:
        assert result == [{"x": (1.0, 2.0), "y": (3.0, 4.0), "z": 3.0}]
    else:
        assert result == []


def test_read_floor_rectangles_with_empty_heights_returns_empty_list(tmp_path):
    path = _write_fds(
        tmp_path,
        """
        &OBST XB=0.0,2.0,0.0,2.0,0.0,3.0 /
        """,
    )

    result = helpers.read_floor_rectangles(str(path), heights=[])

    assert result == []
