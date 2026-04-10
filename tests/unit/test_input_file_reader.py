from simvue_fds.connector import FDSRun
import pytest
import pathlib


@pytest.mark.parametrize("file_name", ["fds_input.fds", "fds_input_pyrosim.fds"])
def test_valid_input_file(file_name: str):
    file_path = pathlib.Path(__file__).parent.joinpath(
        "example_data", "fds_inputs", file_name
    )
    _, fds_content = FDSRun(mode="disabled")._input_file_parser(file_path)

    # Check HEAD information is present
    assert fds_content["head"]["chid"] == "fds_test"

    # Check other lines are present
    assert fds_content["misc"]["maximum_visibility"] == 10
    assert fds_content["mesh"]["xb"] == [0, 3, 0, 4, 0, 3]
    assert fds_content["time"]["t_end"] == 5
    assert fds_content["dump"]["nframes"] == 5
    assert fds_content["reac"]["id"] == "propane reaction"
    assert fds_content["obst"]["surf_id"] == "fire1"
    assert fds_content["surf"]["id"] == "fire1"
    assert fds_content["slcf"]["quantity"] == "TEMPERATURE"

    # Check both DEVC devices created
    assert fds_content["_grp_devc_0"]["quantity"] == "VISIBILITY"
    assert fds_content["_grp_devc_1"]["quantity"] == "TEMPERATURE"


@pytest.mark.parametrize(
    "file_name", ["fds_input_invalid.fds", "fds_input_no_head.fds"]
)
def test_invalid_input_file(file_name: str):
    file_path = pathlib.Path(__file__).parent.joinpath(
        "example_data", "fds_inputs", file_name
    )
    _, fds_content = FDSRun(mode="disabled")._input_file_parser(file_path)

    assert not fds_content
