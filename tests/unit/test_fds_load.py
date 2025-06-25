
import pytest
import tempfile
import pathlib
from unittest.mock import patch
from simvue_fds.connector import FDSRun
import shutil
def do_nothing(self, *_, **__):
    return

def store_metadata(self, metadata):
    self.metadata = metadata
    return

@patch.object(FDSRun, 'init', do_nothing)
@patch.object(FDSRun, '_post_simulation', do_nothing)
@patch.object(FDSRun, 'save_file', do_nothing)
@patch.object(FDSRun, '_tidy_run', do_nothing)
@patch.object(FDSRun, 'update_metadata', store_metadata)
def test_chid_from_input_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copy(pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"), pathlib.Path(temp_dir).joinpath("test_chid.fds"))
        pathlib.Path(temp_dir).joinpath("fds_test_1.sf.bnd").touch()
    
        with FDSRun() as run:
            run.init()
            run.load(temp_dir)
            
            assert run._chid == "fds_test"
            # Confirm that metadata parsed from input file and uploaded
            assert run.metadata["input_file"]["head"]["chid"] == "fds_test"


@patch.object(FDSRun, 'init', do_nothing)
@patch.object(FDSRun, '_post_simulation', do_nothing)
@patch.object(FDSRun, 'save_file', do_nothing)
@patch.object(FDSRun, '_tidy_run', do_nothing)
@patch.object(FDSRun, 'update_metadata', store_metadata)
def test_chid_from_results_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        pathlib.Path(temp_dir).joinpath("test_chid_1.sf.bnd").touch()
        pathlib.Path(temp_dir).joinpath("test_chid.smv").touch()
        pathlib.Path(temp_dir).joinpath("test_chid_steps.csv").touch()
        with FDSRun() as run:
            run.init()
            run.load(temp_dir)
            
            assert run._chid == "test_chid"

@patch.object(FDSRun, 'init', do_nothing)
@patch.object(FDSRun, '_post_simulation', do_nothing)
@patch.object(FDSRun, 'save_file', do_nothing)
@patch.object(FDSRun, '_tidy_run', do_nothing)
@patch.object(FDSRun, 'update_metadata', store_metadata)
def test_chid_invalid_results_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        pathlib.Path(temp_dir).joinpath("test_chid_1.sf.bnd").touch()
        pathlib.Path(temp_dir).joinpath("fds_test.smv").touch()
        pathlib.Path(temp_dir).joinpath("test_chid_steps.csv").touch()
        with FDSRun() as run:
            run.init()
            
            with pytest.raises(ValueError, match="Could not determine CHID from results directory due to files with inconsistent names."):
                run.load(temp_dir)
                
        