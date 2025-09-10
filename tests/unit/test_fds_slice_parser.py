import pytest
import pathlib
from simvue_fds.connector import FDSRun
import simvue
import time
import numpy
import requests

from unittest.mock import patch
def mock_fds_process(self, *_, **__):
    self.slice_parse_interval = 0.05
    return

def mock_during_sim(self, *_, **__):
    time.sleep(5)
    self._trigger.set()
    return

def mock_post_sim(self, *_, **__):
    return
# Ran an FDS case and stopped it abruptly sometime after 25s simulation time
# This is to simulate the parser reading slice files at ragular intervals
# Test with single and multi meshes
# Test with visibility and temperature
# Test with ignore and dont ignore zeros (floating obstruction cube at x,y,z = 1.25 to 1.75)
@pytest.mark.parametrize("results_path", ("slice_singlemesh", "slice_multimesh"), ids=("single_mesh", "multi_mesh"))
@pytest.mark.parametrize("slice_parameter", ("SOOT VISIBILITY", "TEMPERATURE", None), ids=("visibility", "temperature", "disabled"))
@pytest.mark.parametrize("ignore_zeros", (True, False), ids=("ignore_zeros", "include_zeros"))
@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))     
@patch.object(FDSRun, 'add_process', mock_fds_process)
@patch.object(FDSRun, '_find_fds_executable', lambda _: None)
@patch.object(FDSRun, '_during_simulation', mock_during_sim)
@patch.object(FDSRun, '_post_simulation', mock_post_sim)
def test_fds_slice_parser(folder_setup, results_path, slice_parameter, ignore_zeros, load):
    _prefix = slice_parameter.replace(' ', '_').lower() if slice_parameter else None
    with FDSRun() as run:
        run.init(f"testing_{results_path}_{_prefix}_{'ignore_zeros' if ignore_zeros else 'include_zeros'}_{'load' if load else 'launch'}", folder=folder_setup)
        run_id = run.id
        if load:
            if slice_parameter:
                run.load(
                    pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                    slice_parse_quantity = slice_parameter,
                    slice_parse_ignore_zeros = ignore_zeros
                )
            else:
                run.load(
                    pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                )
        else:
            if slice_parameter:
                run.launch(
                    fds_input_file_path= pathlib.Path(__file__).parent.joinpath("example_data", results_path, "no_vents.fds"),
                    workdir_path = pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                    slice_parse_quantity = slice_parameter,
                    slice_parse_interval = 1,
                    slice_parse_ignore_zeros = ignore_zeros,
                )
            else:
                run.launch(
                    fds_input_file_path= pathlib.Path(__file__).parent.joinpath("example_data", results_path, "no_vents.fds"),
                    workdir_path = pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                )
                
        time.sleep(2)
    
        client = simvue.Client()
        
        metrics_names = [item for item in client.get_metrics_names(run_id)]
        
        # Check all 4 metrics from slice have been created
        expected_metrics = [f"{_prefix}.x.1_5", f"{_prefix}.y.1_5", f"{_prefix}.z.1_5", f"{_prefix}.z.2_5"]
        
        # Mesh is 30x40x30 cells, this means 31x41x31 grid points
        # The data is then transposed so that it is in (row, col) which is what multidimensional metrics expects
        # This means the shapes below are flippedm eg for a slice at fixed x:
            # y dimension = 41
            # z dimension = 31
            # Shape in geometric notation = (41, 31)
            # Shape in row, col notation = (31, 41)
        expected_slice_dims = [(31, 41), (31, 31), (41, 31), (41, 31)]
        
        # Check at least 25 times recorded (since we stopped the sim at 25 - 26s)
        for metric, slice_dims in zip(expected_metrics, expected_slice_dims):
            if not slice_parameter:
                assert metric+".max" not in metrics_names
                assert metric+".min" not in metrics_names
                assert metric+".avg" not in metrics_names
            else:
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
                
                # If including zeros, check any slices at 1.5 have min visibility of zero due to obstruction
                if metric.endswith("1_5") and slice_parameter == "SOOT VISIBILITY" and not ignore_zeros:
                    assert numpy.all(_min == 0)
                # If ignoring zeros, check there are none present
                # Otherwise, z = 2.5 slice doesn't pass through obstruction, so should have no zeros in any case
                else:
                    assert numpy.all(_min != 0)
                    
                # Check that the average visibility is decreasing, or temperature increasing over time
                # Will compare 5 steps apart to allow for outliers and noise
                if slice_parameter == "SOOT VISIBILITY":
                    assert numpy.all(_avg[5:] < _avg[:-5])
                else:
                    assert numpy.all(_avg[5:] > _avg[:-5])
        
                # Check multidimensional metrics are present
                # TODO: Temporary solution since client mehods for multi-d metrics not yet available
                # Check step at 0 (start), 10 (middle), and 25 (end) exists
                for i in (0, 10, 25):
                    response = requests.get(
                        url=f"{run._user_config.server.url}/runs/{run.id}/metrics/{metric}/values?step=1",
                        headers=run._sv_obj._headers,
                    )
                    assert response.status_code == 200   
                    assert numpy.array(response.json().get("array")).shape == slice_dims
            
            