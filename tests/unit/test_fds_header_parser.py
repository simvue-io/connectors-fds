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
def mock_fds_process(self, *_, **__):
    """
    Mock process for writing FDS log header, all at once.
    """
    def create_header(self):
        shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_header.txt"), pathlib.Path(self.workdir_path).joinpath(f"fds_test.out"))
        time.sleep(1)
        self._trigger.set()
        return
    thread = threading.Thread(target=create_header, args=(self,))
    thread.start()
    
@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))     
@patch.object(FDSRun, 'add_process', mock_fds_process)
def test_fds_header_parser(folder_setup, load):
    """
    Check that metadata from the header of the log file is correctly uploaded.
    """
    name = 'test_fds_header_parser-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        if load:
            shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"), pathlib.Path(temp_dir.name).joinpath("fds_input.fds"))
            shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_header.txt"), pathlib.Path(temp_dir.name).joinpath("fds_test.out"))
            run.load(
                results_dir = temp_dir.name,
            )
        else:
            run.launch(
                fds_input_file_path = pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
                workdir_path = temp_dir.name,
            )
        
    client = simvue.Client()
    
    run_data = client.get_run(run_id).metadata
    assert run_data["fds"]["revision"] == "FDS-6.9.1-0-g889da6a-release"        
    assert run_data["fds"]["revision_date"] == "Sun Apr 7 17:05:06 2024 -0400"
    assert run_data["fds"]["compiler"] == "Intel(R) Fortran Intel(R) 64 Compiler Classic for applications running on Intel(R) 64, Version 2021.7.1 Build 20221019_000000"
    assert run_data["fds"]["compilation_date"] == "Apr 09, 2024 13:24:51"
    assert run_data["fds"]["mpi_processes"] == '1'
    assert run_data["fds"]["mpi_version"] == '3.1'
    assert run_data["fds"]["mpi_library_version"] == "Intel(R) MPI Library 2021.6 for Linux* OS"