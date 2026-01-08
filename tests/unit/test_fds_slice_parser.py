import pytest
import pathlib
from simvue_fds.connector import FDSRun
import simvue
import time
import numpy
import requests

from unittest.mock import patch
def mock_fds_process(self, *_, **__):
    return

def mock_during_sim(self, *_, **__):
    time.sleep(5)
    self._trigger.set()
    return

def mock_post_sim(self, *_, **__):
    if self.slice_parser:
        self.slice_parser.join()
    time.sleep(2)
    return
# Ran an FDS case and stopped it abruptly sometime after 25s simulation time
# This is to simulate the parser reading slice files at ragular intervals
# Test with single and multi meshes
# Test with visibility and temperature
@pytest.mark.parametrize("results_path", ("slice_singlemesh", "slice_multimesh"), ids=("single_mesh", "multi_mesh"))
@pytest.mark.parametrize("slice_parameter", ("SOOT VISIBILITY", ["SOOT VISIBILITY", "TEMPERATURE"], None), ids=("visibility", "visibility-temperature", "no-quantities"))
@pytest.mark.parametrize("slice_ids", (["temperature_slice", "visibility_slice", "velocity_slice"], None), ids=("ids", "no-ids"))
@pytest.mark.parametrize("enabled", (True, False), ids=("enabled", "disabled"))    
@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))     
@patch.object(FDSRun, 'add_process', mock_fds_process)
@patch.object(FDSRun, '_find_fds_executable', lambda _: None)
@patch.object(FDSRun, '_during_simulation', mock_during_sim)
@patch.object(FDSRun, '_post_simulation', mock_post_sim)
def test_fds_slice_parser(folder_setup, results_path, slice_parameter, slice_ids, enabled, load):
    with FDSRun() as run:
        run.config(disable_resources_metrics = True)
        run.init(f"testing_{results_path}_{'enabled' if enabled else 'disabled'}_{'quantities' if slice_parameter else 'no-quantities'}_{'ids' if slice_ids else 'no-ids'}_{'load' if load else 'launch'}", folder=folder_setup)
        run_id = run.id
        if load:
            if type(slice_parameter) is str:
                run.load(
                    pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                    slice_parse_quantity = slice_parameter,
                    slice_parse_ids = slice_ids,
                    
                )
            else:
                run.load(
                    pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                    slice_parse_enabled = enabled,
                    slice_parse_quantities = slice_parameter,
                    slice_parse_ids = slice_ids,
                )
        else:
            if type(slice_parameter) is str:
                run.launch(
                    fds_input_file_path= pathlib.Path(__file__).parent.joinpath("example_data", results_path, "no_vents.fds"),
                    workdir_path = pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                    slice_parse_quantity = slice_parameter,
                    slice_parse_ids = slice_ids,
                    slice_parse_interval = 3,
                )
            else:
                run.launch(
                    fds_input_file_path= pathlib.Path(__file__).parent.joinpath("example_data", results_path, "no_vents.fds"),
                    workdir_path = pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                    slice_parse_enabled = enabled,
                    slice_parse_quantities = slice_parameter,
                    slice_parse_ids = slice_ids,
                    slice_parse_interval = 3
                )
                
        # Mesh is 30x40x30 cells, this means 31x41x31 grid points
        # The data is then transposed so that it is in (row, col) which is what multidimensional metrics expects
        # This means the shapes below are flipped, eg for a slice at fixed x:
            # y dimension = 41
            # z dimension = 31
            # Shape in geometric notation = (41, 31)
            # Shape in row, col notation = (31, 41)
        _slice_dims = [(31, 41), (31, 31), (41, 31), (41, 31)]
                    
        # Check all metrics from slice have been created
        if type(slice_parameter) is str:
            # This is the deprecated way to specify - automatically enables the feature
            expected_metrics = ["visibility_slice"] if slice_ids else ["soot_visibility.x.1_5", "visibility_slice", "soot_visibility.z.1_5", "soot_visibility.z.2_5"]
            expected_slice_dims = _slice_dims[1:2] if slice_ids else _slice_dims
        elif slice_parameter:
            if not enabled:
                expected_metrics = None
            else:
                expected_metrics = ["temperature_slice", "visibility_slice"] if slice_ids else ["temperature.x.1_5", "temperature_slice", "temperature.z.1_5", "temperature.z.2_5", "soot_visibility.x.1_5", "visibility_slice", "soot_visibility.z.1_5", "soot_visibility.z.2_5"]
                expected_slice_dims = _slice_dims[1:2] * 2 if slice_ids else _slice_dims * 2
        else:
            if not enabled:
                expected_metrics = None
            else:
                expected_metrics = ["temperature_slice", "visibility_slice", "velocity_slice"] if slice_ids else ["temperature.x.1_5", "temperature_slice", "temperature.z.1_5", "temperature.z.2_5", "soot_visibility.x.1_5", "visibility_slice", "soot_visibility.z.1_5", "soot_visibility.z.2_5", "velocity_slice"]
                expected_slice_dims = _slice_dims[1:2] * 3 if slice_ids else _slice_dims * 2 + _slice_dims[1:2]
           
        client = simvue.Client()
        
        metrics_names = [item for item in client.get_metrics_names(run_id)]
        
        if not expected_metrics:
            assert not metrics_names
        else:
            # Check at least 25 times recorded (since we stopped the sim at 25 - 26s)

            for metric, slice_dims in zip(expected_metrics, expected_slice_dims):
                    assert metric+".max" in metrics_names
                    assert metric+".min" in metrics_names
                    assert metric+".avg" in metrics_names
                    _retrieved = client.get_metric_values(run_ids=[run_id], metric_names = [metric+".max", metric+".min", metric+".avg"], xaxis="time")
                    _max = numpy.array(list(_retrieved[f"{metric+'.max'}"].values()))
                    _min = numpy.array(list(_retrieved[f"{metric+'.min'}"].values()))
                    _avg = numpy.array(list(_retrieved[f"{metric+'.avg'}"].values()))
                    
                    # Each measurement should have >= 25, < 30 entries
                    # Since we manually stopped the FDS run at this point
                    assert len(_max) >= 25 and len(_max) < 30
                    assert len(_min) >= 25 and len(_min) < 30
                    assert len(_avg) >= 25 and len(_avg) < 30
                    
                    # Check all max >= avg >= min
                    assert numpy.all(_max >= _avg)
                    assert numpy.all(_avg >= _min)
                        
                    # Check that the average visibility is decreasing, or temperature increasing over time
                    # Will compare 5 steps apart to allow for outliers and noise
                    if "visibility" in metric:
                        assert numpy.all(_avg[5:] < _avg[:-5])
                    elif "temperature" in metric:
                        assert numpy.all(_avg[5:] > _avg[:-5])
            
                    # Check multidimensional metrics are present
                    # TODO: Temporary solution since client mehods for multi-d metrics not yet available
                    # Check step at 0 (start), 10 (middle), and 25 (end) exists
                    time.sleep(1)
                    for i in (0, 10, 25):
                        response = requests.get(
                            url=f"{run._user_config.server.url}/runs/{run.id}/metrics/{metric}/values?step={i}",
                            headers=run._sv_obj._headers,
                        )
                        assert response.status_code == 200   
                        assert numpy.array(response.json().get("array")).shape == slice_dims
                
            