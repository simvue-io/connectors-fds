from simvue_fds.connector import FDSRun
import simvue
import threading
import time
import tempfile
from unittest.mock import patch
import uuid
import pathlib
import pandas
import pytest
import shutil
import numpy
import datetime

def write_to_log(self, example_file, temp_logfile):
    for line in example_file:
        temp_logfile.write(line)
        time.sleep(0.1)
    temp_logfile.flush()
    time.sleep(1)
    temp_logfile.close()
    self._trigger.set()
    return

def mock_devc_process(self, *_, **__):
    """
    Mock process for creating DEVC CSV output files, written line by line
    """
    temp_logfile = pathlib.Path(self.workdir_path).joinpath("fds_test_devc.csv").open(mode="w", buffering=1)
    example_file = pathlib.Path(__file__).parent.joinpath("example_data", "fds_devc.csv").open("r")
    thread = threading.Thread(target=write_to_log, args=(self, example_file, temp_logfile))
    thread.start()
    
def mock_hrr_process(self, *_, **__):
    """
    Mock process for creating HRR CSV output files, written line by line
    """
    temp_logfile = pathlib.Path(self.workdir_path).joinpath("fds_test_hrr.csv").open(mode="w", buffering=1)
    example_file = pathlib.Path(__file__).parent.joinpath("example_data", "fds_hrr.csv").open("r")
    thread = threading.Thread(target=write_to_log, args=(self, example_file, temp_logfile))
    thread.start()

def mock_ctrl_process(self, *_, **__):
    """
    Mock process for creating CTRL log CSV output files, written line by line
    """
    temp_logfile = pathlib.Path(self.workdir_path).joinpath("fds_test_devc_ctrl_log.csv").open(mode="w", buffering=1)
    example_file = pathlib.Path(__file__).parent.joinpath("example_data", "fds_devc_ctrl_log.csv").open("r")
    thread = threading.Thread(target=write_to_log, args=(self, example_file, temp_logfile))
    thread.start()

@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))
@patch.object(FDSRun, 'add_process', mock_devc_process)
def test_fds_devc_parser(folder_setup, load):
    """
    Check that values of each DEVC device at every timestep are uploaded to Simvue as metrics
    """
    name = 'test_fds_devc_parser-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        if load:
            shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"), pathlib.Path(temp_dir.name).joinpath("fds_input.fds"))
            shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_devc.csv"), pathlib.Path(temp_dir.name).joinpath("fds_test_devc.csv"))
            run.load(
                results_dir = temp_dir.name,
            )
        else:
            run.launch(
                fds_input_file_path = pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
                workdir_path = temp_dir.name,
            )
           
    client = simvue.Client()
        
    # Check that 9 metrics have been created, one for each metric in the CSV
    metrics_names = client.get_metrics_names(run_id)
    assert sum(1 for name in metrics_names) == 9
    
    # Get all metrics from run, check last value of each matches last set of lines in file
    metrics = dict(client.get_run(run_id).metrics)
    csvFile = pandas.read_csv(pathlib.Path(__file__).parent.joinpath("example_data", "fds_devc.csv"))

    expected_metric_names = csvFile.iloc[0][1:].values.tolist()
    expected_metric_names.sort()
    expected_metric_last_values = csvFile.iloc[-1][1:].values.tolist()
    expected_metric_last_values = [float(val) for val in expected_metric_last_values]
    expected_metric_last_values.sort()
      
    actual_metric_names = [key for key in metrics.keys()]
    actual_metric_names.sort()
    actual_metric_last_values = [value["last"] for value in metrics.values()]
    actual_metric_last_values.sort()
        
    assert expected_metric_names == actual_metric_names
    assert expected_metric_last_values == actual_metric_last_values    
    
@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))
@patch.object(FDSRun, 'add_process', mock_hrr_process)
def test_fds_hrr_parser(folder_setup, load):    
    """
    Check that values about Heat Release Rate at every timestep are uploaded to Simvue as metrics
    """
    name = 'test_fds_hrr_parser-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        if load:
            shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"), pathlib.Path(temp_dir.name).joinpath("fds_input.fds"))
            shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_hrr.csv"), pathlib.Path(temp_dir.name).joinpath("fds_test_hrr.csv"))
            run.load(
                results_dir = temp_dir.name,
            )
        else:
            run.launch(
                fds_input_file_path = pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
                workdir_path = temp_dir.name,
            )
           
    client = simvue.Client()
        
    # Check that 16 metrics have been created, one for each metric in the CSV
    metrics_names = client.get_metrics_names(run_id)
    assert sum(1 for name in metrics_names) == 16
    
    # Get all metrics from run, check last value of each matches last set of lines in file
    metrics = dict(client.get_run(run_id).metrics)
    csvFile = pandas.read_csv(pathlib.Path(__file__).parent.joinpath("example_data", "fds_hrr.csv"))

    expected_metric_names = csvFile.iloc[0][1:].values.tolist()
    expected_metric_names.sort()
    expected_metric_last_values = csvFile.iloc[-1][1:].values.tolist()
    expected_metric_last_values = [float(val) for val in expected_metric_last_values]
    expected_metric_last_values.sort()
    
    actual_metric_names = [key for key in metrics.keys()]
    actual_metric_names.sort()
    actual_metric_last_values = [value["last"] for value in metrics.values()]
    actual_metric_last_values.sort()
        
    assert expected_metric_names == actual_metric_names
    assert expected_metric_last_values == actual_metric_last_values
    
@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))
@patch.object(FDSRun, 'add_process', mock_ctrl_process)
def test_fds_ctrl_parser(folder_setup, load):    
    """
    Check that activation status of CTRL and DEVC devices are written to Events log and metadata.
    """
    name = 'test_fds_ctrl_parser-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name,folder=folder_setup)
        run_id = run.id
        if load:
            shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"), pathlib.Path(temp_dir.name).joinpath("fds_input.fds"))
            shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_devc_ctrl_log.csv"), pathlib.Path(temp_dir.name).joinpath("fds_test_devc_ctrl_log.csv"))
            run.load(
                results_dir = temp_dir.name,
            )
        else:
            run.launch(
                fds_input_file_path = pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
                workdir_path = temp_dir.name,
            )
           
    client = simvue.Client()
    
    # Check DEVC and CTRL events have been correctly added to events log
    events = [event['message'] for event in client.get_events(run_id)]
    assert "DEVC 'Ceiling_Thermocouple.Back_Right' has been set to 'True' at time 4.25244E+01s, when it reached a value of 2.00162E+02C." in events
    assert "CTRL 'KILL_TEMP_TOO_HIGH' has been set to 'True' at time 4.25244E+01s" in events
    
    # Check metadata has been added correctly
    metadata = client.get_run(run_id).metadata
    assert metadata.get('Ceiling_Thermocouple.Back_Right') == True
    assert metadata.get('KILL_TEMP_TOO_HIGH') == True
