from simvue_fds.connector import FDSRun
import simvue
import pathlib
import shutil
import time
import uuid
import tempfile
import threading
from unittest.mock import patch
import pytest
import shutil
import f90nml

def parse_line_data(run, input_path):
        # Load input file as data (normally done by Multiparser)
        nml = f90nml.read(input_path).todict()
        run._input_dict = {k: v for k, v in nml.items() if v}
        
        # Add this which is typically added in launch()
        run._line_var_coords = {}
        
        # Generate mapping
        run._map_line_var_coords()

def test_single_devc_xb():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "single_devc_xb.fds"))
        
        # Check device ID has been mapped to correct axis label (default: temp_line-y)
        assert run._line_var_coords["temp_line"] == "temp_line-y"
        
def test_single_devc_xbp():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "single_devc_xbp.fds"))
        
        # Check device ID has been mapped to correct axis label (default: temp_line-y)
        assert run._line_var_coords["temp_line"] == "temp_line-y"
        
def test_single_devc_coord_id():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "single_devc_coord_id.fds"))
        
        # Check device ID has been mapped to correct axis label (user specified: y_coords)
        assert run._line_var_coords["temp_line"] == "y_coords"
        
        
def test_single_devc_multi_dimension():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "single_devc_multi_dimension.fds"))
        
        # Only supports 1D line devices for now - so this should report None, indicating that this device should be skipped when adding metrics
        assert run._line_var_coords["temp_line"] == None
        
def test_single_devc_no_id():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "single_devc_no_id.fds"))
        
        # Should skip devices which have no ID provided
        assert run._line_var_coords == {}
        
def test_multi_devc():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "multi_devc.fds"))
        
        # Check both devices mapped to their default values
        assert run._line_var_coords["temp_line_y"] == "temp_line_y-y"
        assert run._line_var_coords["temp_line_x"] == "temp_line_x-x"
        
def test_multi_devc_coord_id():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "multi_devc_coord_id.fds"))
        
        # Check both devices mapped to their user specified values
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["temp_line_x"] == "x_coords"
        
def test_multi_devc_hide_coords():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "multi_devc_hide_coords.fds"))
        
        # Check both devices are using default coord name from first defined device
        assert run._line_var_coords["temp_line_y"] == "temp_line_y-y"
        assert run._line_var_coords["velocity_line_y"] == "temp_line_y-y"
        
def test_multi_devc_coord_id_hide_coords():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "multi_devc_coord_id_hide_coords.fds"))
        
        # Check both devices are using user specified coord name from first defined device
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["velocity_line_y"] == "y_coords"
        
def test_multi_devc_multi_hide_coords():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "multi_devc_multi_hide_coords.fds"))
        
        # Check first two devices are using user specified 'y_coords'
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["velocity_line_y"] == "y_coords"
        
        # Check second two devices are using default z coords
        assert run._line_var_coords["temp_line_z"] == "temp_line_z-z"
        assert run._line_var_coords["velocity_line_z"] == "temp_line_z-z"
        
def test_multi_devc_hide_coords_invalid():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "multi_devc_hide_coords_invalid.fds"))
        
        # Check first two devices are using user specified 'y_coords'
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["velocity_line_y"] == "y_coords"
        
        # Check last two devices are using default z coords
        assert run._line_var_coords["temp_line_z"] == "temp_line_z-z"
        assert run._line_var_coords["velocity_line_z"] == "temp_line_z-z"
        
        # Check middle two devices are None, since they have 2D axes which we do not support
        assert run._line_var_coords["temp_line_multi_dim"] == None
        assert run._line_var_coords["temp_line_multi_dim"] == None
        
        
def test_multi_point_and_line_devc():
    with FDSRun() as run:
        parse_line_data(run, pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", "multi_point_and_line_devc.fds"))
        
        # Check point DEVC devices are not in mapping
        assert not run._line_var_coords.get("velocity_point")
        assert not run._line_var_coords.get("temperature_point")
        assert not run._line_var_coords.get("visibility_point")
        
        # Check line DEVC devices are correctly using user specified 'y_coords'
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["velocity_line_y"] == "y_coords"