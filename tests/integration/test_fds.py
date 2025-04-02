from examples.fds_example import fds_example
from examples.load_historic_runs import load_runs_example
import pytest
import subprocess
import pathlib
import tempfile
import simvue
from simvue.sender import sender

@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))
@pytest.mark.parametrize("parallel", (True, False), ids=("parallel", "serial"))
@pytest.mark.parametrize("offline", (True, False), ids=("offline", "online"))
def test_fds_connector(folder_setup, load, offline, parallel):
    
    if load:
        if parallel:
            pytest.skip("Parallel has no effect when loading from historic runs")
        else:
            run_id = load_runs_example(folder_setup, offline)
    else:
        try:
            subprocess.run("fds")
        except FileNotFoundError:
            pytest.skip("You are attempting to run FDS Integration Tests without having FDS installed in your path.")
        
        run_id = fds_example(folder_setup, offline, parallel)

    if offline:
        _id_mapping = sender()
        run_id = _id_mapping.get(run_id)
    
    client = simvue.Client()
    run_data = client.get_run(run_id)
    events = [event["message"] for event in client.get_events(run_id)]
    
    # Check run description and tags from init have been added
    assert run_data.description == "An example of using the FDSRun Connector to track an FDS simulation."
    assert run_data.tags == ['fds', 'vents']
    
    # Check alert has been added
    assert "visibility_below_three_metres" in [alert["name"] for alert in run_data.get_alert_details()]
    
    # Check metadata from header
    if parallel:
        assert run_data.metadata["fds"]["mpi_processes"] == '2'
    else:
        assert run_data.metadata["fds"]["mpi_processes"] == '1'
        
    # Check metadata from input file
    assert run_data.metadata["input_file"]["_grp_devc_1"]["id"] == "flow_volume_supply"
    
    # Check events from log
    assert "Time Step: 1, Simulation Time: 0.092 s" in events
    
    # Check events from DEVC/CTRL log
    assert "DEVC 'timer' has been set to 'True' at time 2.00097E+00s, when it reached a value of 2.00097E+00s." in events
    
    metrics = dict(run_data.metrics)
    # Check metrics from HRR file
    assert metrics["HRR"]["count"] > 0
    
    # Check metrics from DEVC file
    assert metrics["flow_volume_supply"]["count"] > 0
    
    # Check metrics from log file
    assert metrics["max_pressure_error"]["count"] > 0
    assert metrics["max_divergence.mesh.2"]["count"] > 0

    temp_dir = tempfile.TemporaryDirectory()
    
    # Check input file uploaded as input
    client.get_artifacts_as_files(run_id, "input", temp_dir.name)
    assert pathlib.Path(temp_dir.name).joinpath("activate_vents.fds").exists()
    
    # Check results uploaded as output
    client.get_artifacts_as_files(run_id, "output", temp_dir.name)
    assert pathlib.Path(temp_dir.name).joinpath("supply_exhaust_vents.smv").exists()