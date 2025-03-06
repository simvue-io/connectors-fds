from simvue_fds.connector import FDSRun
import simvue
import threading
import time
import tempfile
from unittest.mock import patch
import uuid
import pathlib
import pytest

testdata = [
    (
        "fds_log.txt",
        1,
        {
            "max_vn": 0.14,
            "max_divergence": 3.2,
            "min_divergence": -1.0,
            "max_cfl": 0.17,
            "max_pressure_error": 730,
            "total_heat_release_rate": 152.324,
            "max_velocity_error": 0.0087,
            "radiation_loss_to_boundaries": -35.118,
            "pressure_iteration": 1
        },
    ),
    (
        "fds_log_multimesh.txt",
        2,
        {
            "max_cfl.mesh.1": 0.79,
            "max_divergence.mesh.1": 2.8,
            "min_divergence.mesh.1": -0.93,
            "max_vn.mesh.1": 0.60,
            "total_heat_release_rate.mesh.1": 71.663,
            "radiation_loss_to_boundaries.mesh.1": -18.336,
            "pressure_iteration": 2,
            "max_velocity_error": 0.022,
            "max_pressure_error": 13,
            "max_cfl.mesh.2": 0.77,
            "max_divergence.mesh.2": 2.8,
            "min_divergence.mesh.2": -0.70,
            "max_vn.mesh.2": 0.35,
            "total_heat_release_rate.mesh.2": 73.372,
            "radiation_loss_to_boundaries.mesh.2": -18.070,
        },
    ),
]

def mock_fds_process(self, *_, **__):
    """
    Mock process for creating FDS log file, in blocks of lines corresponding to each timestep.
    """
    temp_logfile = pathlib.Path(self.workdir_path).joinpath("fds_test.out").open(mode="w")
    def write_to_log():
        log_file = pathlib.Path(__file__).parent.joinpath("example_data", self.example_file).open("r")
        line_counter = 0
        for line in log_file:
            temp_logfile.write(line)
            time.sleep(0.01)
            # FDS file writes to the file in blocks, writing all lines for a given time step at once
            # So keep writing until you find an empty line which marks the end of a block, and flush
            if not line.strip():
                temp_logfile.flush()
            
        temp_logfile.flush()
        time.sleep(1)
        temp_logfile.close()
        self._trigger.set()
        return
    thread = threading.Thread(target=write_to_log)
    thread.start()
    
@pytest.mark.parametrize("file_name,num_meshes,expected_results", testdata, ids=("single_mesh", "multi_mesh"))
@patch.object(FDSRun, 'add_process', mock_fds_process)
def test_fds_log_parser(folder_setup, file_name, num_meshes, expected_results):
    """
    Check that values from FDS log file are uploaded as metrics.
    """ 
    name = 'test_fds_log_parser-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    with FDSRun() as run:
        # Attach example file and number of lines per block - cant find a better way to do pass this into the mocker :/
        run.example_file = file_name
        run.num_meshes = num_meshes
        
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        run.launch(
            fds_input_file_path = pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
            workdir_path = temp_dir.name,
        )
        
           
    client = simvue.Client()
        
    # Check that 9 metrics have been created, one for each line in log per timestep
    metrics_names = client.get_metrics_names(run_id)
    assert sum(1 for name in metrics_names) == len(expected_results.keys())
    
    # Get all metrics from run, check last value of each matches last set of lines in file
    run_data = client.get_run(run_id)
    metrics = dict(run_data.metrics)
    for key, values in metrics.items():
        assert values["last"] == expected_results[key]
        
    # Check device activation time added as metadata
    assert run_data.metadata.get("timer_activation_time") == 3.003