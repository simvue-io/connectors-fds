from simvue_fds.connector import FDSRun
import pathlib
from unittest.mock import patch
import tempfile
import uuid
import simvue
import filecmp
import shutil
import time
import threading

def mock_fds_process(self, *_, **__):
    """
    Mock process - does nothing but set termination trigger
    """
    time.sleep(1)
    self._trigger.set()
    return True

@patch.object(FDSRun, '_find_fds_executable', lambda _: None)
@patch.object(FDSRun, 'add_process', mock_fds_process)
def test_fds_file_deletion(folder_setup):    
    """
    Check that files beginning with CHID are deleted from workdir.
    """    
    name = 'test_fds_file_deletion-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    
    # Copy results into workdir
    shutil.copytree(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), temp_dir.name, dirs_exist_ok=True)
    # Create a file which doesn't start with chid - shouldn't ever be deleted
    pathlib.Path(temp_dir.name).joinpath("test.txt").touch()
    # Copy in an input file - also shouldn't be deleted
    shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"), temp_dir.name)
    
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        run.launch(
            fds_input_file_path = pathlib.Path(temp_dir.name).joinpath("fds_input.fds"),
            workdir_path = temp_dir.name,
            clean_workdir = True
        )
        
        comparison = filecmp.dircmp(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), temp_dir.name)
        assert sorted(comparison.left_only) == sorted(["fds_test.smv", "fds_test_1_1.s3d.sz", "fds_test_1_1.sf.bnd"])
        assert sorted(comparison.right_only) == sorted(["test.txt", "fds_input.fds"])
        
@patch.object(FDSRun, '_find_fds_executable', lambda _: None) 
@patch.object(FDSRun, 'add_process', mock_fds_process)
def test_fds_no_file_deletion(folder_setup):    
    """
    Check that no files are deleted if clean is not specified.
    """    
    name = 'test_fds_no_file_deletion-%s' % str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix="fds_test")
    
    # Copy results into workdir
    shutil.copytree(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), temp_dir.name, dirs_exist_ok=True)
    # Create a file which doesn't start with chid - shouldn't ever be deleted
    pathlib.Path(temp_dir.name).joinpath("test.txt").touch()
    # Copy in an input file - also shouldn't be deleted
    shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"), temp_dir.name)
    
    with FDSRun() as run:
        run.config(disable_resources_metrics=True)
        run.init(name=name, folder=folder_setup)
        run_id = run.id
        run.launch(
            fds_input_file_path = pathlib.Path(temp_dir.name).joinpath("fds_input.fds"),
            workdir_path = temp_dir.name,
        )
        
        comparison = filecmp.dircmp(pathlib.Path(__file__).parent.joinpath("example_data", "fds_outputs"), temp_dir.name)
        assert not comparison.left_only
        assert sorted(comparison.right_only) == sorted(["test.txt", "fds_input.fds"])