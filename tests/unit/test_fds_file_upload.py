from simvue_integrations_fds.connector import FDSRun
from simvue.api.objects import Run
import pathlib
from unittest.mock import patch, MagicMock
import tempfile
import uuid
import simvue
import filecmp
import shutil
import time
import threading

def mock_fds_process(self, *_, **__):
    """
    Mock process which copies results from example data to tmp.
    """
    shutil.copytree(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), self.workdir_path, dirs_exist_ok=True)    
    time.sleep(1)
    self._trigger.set()
    return True
    
@patch.object(FDSRun, 'add_process', mock_fds_process)
def test_fds_file_upload(folder_setup):    
    """
    Check that all results are uploaded from workdir.
    """
    name = 'test_fds_file_upload-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        run.launch(
            fds_input_file_path = pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
            workdir_path = temp_dir.name
        )
        
        client = simvue.Client()
        
        # Retrieve all outputs from server and check all files exist and are the same
        retrieved_dir = pathlib.Path(temp_dir.name).joinpath("retrieved_results")
        retrieved_dir.mkdir()
        client.get_artifacts_as_files(run_id, "output", str(retrieved_dir))
        comparison = filecmp.dircmp(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), str(retrieved_dir))
        assert not (comparison.diff_files or comparison.left_only or comparison.right_only)
        
@patch.object(FDSRun, 'add_process', mock_fds_process)
def test_fds_file_specified_upload(folder_setup):    
    """
    Check that all results are uploaded from workdir.
    """
    name = 'test_fds_file_specified_upload-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        run.launch(
            fds_input_file_path = pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
            workdir_path = temp_dir.name,
            upload_files = ["fds_test.smv", "fds_test_1_1.s3d.sz"]
        )
        
        client = simvue.Client()
        
        # Retrieve all outputs from server and check all files exist and are the same
        retrieved_dir = pathlib.Path(temp_dir.name).joinpath("retrieved_results")
        retrieved_dir.mkdir()
        client.get_artifacts_as_files(run_id, "output", str(retrieved_dir))
        comparison = filecmp.dircmp(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), str(retrieved_dir))

        assert not (comparison.diff_files or comparison.right_only)
        assert comparison.left_only == ["fds_test_1_1.sf.bnd"]
        
        
def mock_aborted_fds_process(self, *_, **__):
    """
    Mock a long running FDS process which is aborted by the server
    """
    shutil.copytree(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), self.workdir_path, dirs_exist_ok=True)
    
    def aborted_process(self):
        """
        Long running process which should be interrupted at the next heartbeat by the creation of a .stop file
        """
        self._heartbeat_interval = 2
        stop_file = pathlib.Path(self.workdir_path).joinpath("fds_test.stop")
        time_elapsed = 0
        while time_elapsed < 30:
            if stop_file.exists():
                stop_file.unlink()
                break
            time.sleep(1)
            time_elapsed += 1
        self._trigger.set()
        return

        
    thread = threading.Thread(target=aborted_process, args=(self,))
    thread.start()
    
def abort(self):
    """
    Instead of making an API call to the server, just sleep for 1s and return True to indicate an abort has been triggered
    """
    time.sleep(2)
    return True
    
@patch.object(FDSRun, 'add_process', mock_aborted_fds_process)
@patch.object(Run, 'abort_trigger', abort) 
def test_fds_file_upload_after_abort(folder_setup):
    """
    Check that outputs are uploaded if the simulation is aborted early by Simvue
    """    
    name = 'test_fds_file_upload_after_abort-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        run.launch(
            fds_input_file_path = pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
            workdir_path = temp_dir.name
        )
        
        client = simvue.Client()
        
    # Check that run was aborted correctly, and did not exist for longer than 10s
    runtime = client.get_run(run_id).runtime
    assert runtime.tm_sec < 30
    
    # Retrieve all outputs from server and check all files exist and are the same
    retrieved_dir = pathlib.Path(temp_dir.name).joinpath("retrieved_results")
    retrieved_dir.mkdir()
    client.get_artifacts_as_files(run_id, "output", str(retrieved_dir))
    comparison = filecmp.dircmp(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), str(retrieved_dir))
    assert not (comparison.diff_files or comparison.left_only or comparison.right_only)