from simvue_fds.connector import FDSRun
import pathlib
import f90nml
import requests
from simvue.config.user import SimvueConfiguration
def parse_line_data(run, scenario_name):
        # Load input file as data (normally done by Multiparser)
        nml = f90nml.read(pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", f"{scenario_name}.fds")).todict()
        run._input_dict = {k: v for k, v in nml.items() if v}
        
        # Add this which is typically added in launch()
        run._line_var_coords = {}
        
        # Generate mapping
        run._map_line_var_coords()
        
def extract_line_metrics(run, scenario_name):
    meta, data = run._line_parser(str(pathlib.Path(__file__).parent.joinpath("example_data", "line_devices", f"{scenario_name}.csv")))
    run._line_callback(data, meta)
    
def get_line_metrics(run_id, metric):
    # Check slice uploaded as 3D metric
    _user_config: SimvueConfiguration = SimvueConfiguration.fetch()
    response = requests.get(
        url=f"{_user_config.server.url}/runs/{run_id}/metrics/{metric}/values?step=0",
        headers={
            "Authorization": f"Bearer {_user_config.server.token.get_secret_value()}",
            "User-Agent": "Simvue Python client",
            "Accept-Encoding": "gzip",
        }
    )
    assert response.status_code == 200
    return response.json()

def test_single_devc_xb():
    scenario_name = "single_devc_xb"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check device ID has been mapped to correct axis label (default: temp_line-y)
        assert run._line_var_coords["temp_line"] == "temp_line-y"
        
def test_single_devc_xbp():
    scenario_name = "single_devc_xbp"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check device ID has been mapped to correct axis label (default: temp_line-y)
        assert run._line_var_coords["temp_line"] == "temp_line-y"
        
def test_single_devc_coord_id():
    scenario_name = "single_devc_coord_id"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check device ID has been mapped to correct axis label (user specified: y_coords)
        assert run._line_var_coords["temp_line"] == "y_coords"
        
        
def test_single_devc_multi_dimension():
    scenario_name = "single_devc_multi_dimension"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Only supports 1D line devices for now - so this should report None, indicating that this device should be skipped when adding metrics
        assert run._line_var_coords["temp_line"] == None
        
def test_single_devc_no_id():
    scenario_name = "single_devc_no_id"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Should skip devices which have no ID provided
        assert run._line_var_coords == {}
        
def test_multi_devc():
    scenario_name = "multi_devc"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check both devices mapped to their default values
        assert run._line_var_coords["temp_line_y"] == "temp_line_y-y"
        assert run._line_var_coords["temp_line_x"] == "temp_line_x-x"
        
def test_multi_devc_coord_id():
    scenario_name = "multi_devc_coord_id"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check both devices mapped to their user specified values
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["temp_line_x"] == "x_coords"
        
def test_multi_devc_hide_coords():
    scenario_name = "multi_devc_hide_coords"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check both devices are using default coord name from first defined device
        assert run._line_var_coords["temp_line_y"] == "temp_line_y-y"
        assert run._line_var_coords["velocity_line_y"] == "temp_line_y-y"
        
def test_multi_devc_coord_id_hide_coords():
    scenario_name = "multi_devc_coord_id_hide_coords"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check both devices are using user specified coord name from first defined device
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["velocity_line_y"] == "y_coords"
        
def test_multi_devc_multi_hide_coords():
    scenario_name = "multi_devc_multi_hide_coords"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check first two devices are using user specified 'y_coords'
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["velocity_line_y"] == "y_coords"
        
        # Check second two devices are using default z coords
        assert run._line_var_coords["temp_line_z"] == "temp_line_z-z"
        assert run._line_var_coords["velocity_line_z"] == "temp_line_z-z"
        
def test_multi_devc_hide_coords_invalid():
    scenario_name = "multi_devc_hide_coords_invalid"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
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
    scenario_name = "multi_point_and_line_devc"
    with FDSRun() as run:
        parse_line_data(run, scenario_name)
        
        # Check point DEVC devices are not in mapping
        assert not run._line_var_coords.get("velocity_point")
        assert not run._line_var_coords.get("temperature_point")
        assert not run._line_var_coords.get("visibility_point")
        
        # Check line DEVC devices are correctly using user specified 'y_coords'
        assert run._line_var_coords["temp_line_y"] == "y_coords"
        assert run._line_var_coords["velocity_line_y"] == "y_coords"