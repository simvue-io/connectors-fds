import pytest
import pathlib
from simvue_fds.connector import FDSRun
import simvue
import time
import numpy
import requests
import logging

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
@pytest.mark.parametrize("slice_parameter", (["SOOT VISIBILITY", "TEMPERATURE"], None), ids=("visibility-temperature", "no-quantities"))
@pytest.mark.parametrize("slice_fixed_dims", (["x", "y"], None), ids=("x-y", "no-quantities"))
@pytest.mark.parametrize("slice_ids", (["temperature_slice", "visibility_slice", "velocity_slice"], None), ids=("ids", "no-ids"))
@pytest.mark.parametrize("enabled", (True, False), ids=("enabled", "disabled"))    
@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))     
@patch.object(FDSRun, 'add_process', mock_fds_process)
@patch.object(FDSRun, '_find_fds_executable', lambda _: None)
@patch.object(FDSRun, '_during_simulation', mock_during_sim)
@patch.object(FDSRun, '_post_simulation', mock_post_sim)
def test_fds_slice_parser(folder_setup, results_path, slice_parameter, slice_ids, slice_fixed_dims, enabled, load):
    with FDSRun() as run:
        run.config(disable_resources_metrics = True)
        run.init(f"testing_{results_path}_{'enabled' if enabled else 'disabled'}_{'quantities' if slice_parameter else 'no-quantities'}_{'ids' if slice_ids else 'no-ids'}_{'load' if load else 'launch'}", folder=folder_setup)
        run_id = run.id
        if load:
            run.load(
                pathlib.Path(__file__).parent.joinpath("example_data", results_path),
                slice_parse_enabled = enabled,
                slice_parse_quantities = slice_parameter,
                slice_parse_ids = slice_ids,
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
        expected_slices = {
            "temperature.x.1_5": (31, 41),
            "temperature_slice": (31, 31),
            "temperature.z.1_5":  (41, 31),
            "temperature.z.2_5": (41, 31),
            "soot_visibility.x.1_5": (31, 41), 
            "visibility_slice": (31, 31), 
            "soot_visibility.z.1_5":  (41, 31), 
            "soot_visibility.z.2_5": (41, 31), 
            "velocity_slice": (31, 31)
        }
        if slice_parameter:
            expected_slices.pop("velocity_slice", None)
        if slice_fixed_dims:
            expected_slices.pop("soot_visibility.z.1_5", None)
            expected_slices.pop("soot_visibility.z.2_5", None)
            expected_slices.pop("temperature.z.1_5", None)
            expected_slices.pop("temperature.z.2_5", None)
        if slice_ids:
            expected_slices.pop("soot_visibility.z.1_5", None)
            expected_slices.pop("soot_visibility.z.2_5", None)
            expected_slices.pop("soot_visibility.x.1_5", None)
            expected_slices.pop("temperature.z.1_5", None)
            expected_slices.pop("temperature.z.2_5", None)
            expected_slices.pop("temperature.x.1_5", None)
        
        if not enabled:
            expected_slices = None
           
        client = simvue.Client()
        
        metrics_names = [item for item in client.get_metrics_names(run_id)]
        
        if not expected_slices:
            assert not metrics_names
        else:
            # Check at least 25 times recorded (since we stopped the sim at 25 - 26s)
            for metric, slice_dims in expected_slices.items():
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
                
@patch.object(FDSRun, 'add_process', mock_fds_process)
@patch.object(FDSRun, '_find_fds_executable', lambda _: None)
@patch.object(FDSRun, '_during_simulation', mock_during_sim)
@patch.object(FDSRun, '_post_simulation', mock_post_sim)
@pytest.mark.parametrize("slice_id", (None, "cell_centered_slice", "large_slice", "wrong_id"), ids=("all_slices", "cell_centered_slice", "slice_too_big", "wrong_id"))     
def test_fds_invalid_slice(folder_setup, caplog, slice_id):
    with FDSRun() as run:
        run.config(disable_resources_metrics = True)
        run.init(f"test_invalid_slice-{slice_id if slice_id else 'all_slices'}", folder=folder_setup)
        run_id = run.id
        run.launch(
            fds_input_file_path= pathlib.Path(__file__).parent.joinpath("example_data", "slice_invalid", "Pohlhausen_Pr_1.fds"),
            workdir_path = pathlib.Path(__file__).parent.joinpath("example_data", "slice_invalid"),
            slice_parse_enabled = True,
            slice_parse_ids = [slice_id] if slice_id else None,
            slice_parse_interval = 3
        )

    client = simvue.Client()
    events = [event["message"] for event in client.get_events(run_id)]
    metrics_names = [item for item in client.get_metrics_names(run_id)]
    
    if slice_id == "cell_centered_slice" or not slice_id:
        assert all("Unable to parse a slice due to unexpected values within the slice - enable debug logging for full traceback." in s for s in (caplog.text, events))
    if slice_id == "large_slice" or not slice_id:
        assert all("Slice 'large_slice' exceeds the maximum size for upload to the server - ignoring this metric." in s for s in (caplog.text, events))
    
    
    if slice_id:
        assert not metrics_names
    else:
        # Check cell centered slice  and large slice not created
        for metric in ("cell_centered_slice", "large_slice"):
            assert f"{metric}.max" not in metrics_names
            assert f"{metric}.min" not in metrics_names
            assert f"{metric}.avg" not in metrics_names
            
            response = requests.get(
                url=f"{run._user_config.server.url}/runs/{run.id}/metrics/cell_centered_slice/values?step=0",
                headers=run._sv_obj._headers,
            )
            assert response.status_code == 404
        
        # Check other slices are created
        assert "temperature.x.0_0.max" in metrics_names
        assert "temperature.x.0_0.min" in metrics_names
        assert "temperature.x.0_0.avg" in metrics_names
        
        response = requests.get(
            url=f"{run._user_config.server.url}/runs/{run.id}/metrics/temperature.x.0_0/values?step=0",
            headers=run._sv_obj._headers,
        )
        assert response.status_code == 200