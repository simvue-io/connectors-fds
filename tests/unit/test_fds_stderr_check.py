import simvue.executor
from simvue_fds.connector import FDSRun
import simvue
import threading
import time
import tempfile
from unittest.mock import patch
import uuid
import pathlib
import pytest
import subprocess
import sys

# Check it doesnt hang for 60s due to slice parser not finishing
@pytest.mark.timeout(30)
@pytest.mark.parametrize("slice_parse_enabled", (True, False), ids=("slice_parse_enabled", "slice_parse_disabled"))
@pytest.mark.parametrize("file_name", ("fds_invalid_config.stderr", "fds_no_file.stderr", "fds_too_few_meshes.stderr", "fds_expected.stderr"), ids=("invalid_config", "no_file", "cannot_mpi", "expected"))
@patch.object(FDSRun, '_find_fds_executable', lambda _: None)
def test_fds_stderr_check(folder_setup, file_name, slice_parse_enabled):

    def mock_execute_process(*args, file_name=file_name, **kwargs):
        """Execute a process which sleeps for 2s, then passes the stderr from the example file into the completion callback"""
        _result = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
        def mock_process(result=_result, completion_callback=args[3], file_name=file_name):
            while result.poll() is None:
                time.sleep(1)
            std_err = pathlib.Path(__file__).parent.joinpath("example_data", file_name).read_text()
            completion_callback(0, "", std_err)
        thread = threading.Thread(target=mock_process)
        thread.start()
        return _result, thread
    
    with patch("simvue.executor._execute_process", mock_execute_process):
        with tempfile.TemporaryDirectory() as tempd:
            with FDSRun() as run:
                run.config(disable_resources_metrics=True)
                run.init('test_fds_stderr-%s' % str(uuid.uuid4()), folder=folder_setup)
                run_id = run.id
                run.launch(
                    pathlib.Path(__file__).parent.joinpath("example_data", "fds_input.fds"),
                    workdir_path=tempd,
                    slice_parse_enabled=slice_parse_enabled
                    )
            
    time.sleep(1)
    client = simvue.Client()
    run_data = client.get_run(run_id)
    events = [event["message"] for event in client.get_events(run_id)]
    alert = client.get_alerts(run_id=run_id, critical_only=False, names_only=False)[0]
    
    if file_name == "fds_expected.stderr":
        # This is the stuff printed to stderr during a successful FDS run - should not be marked as failed
        assert run_data.status == "completed"
        assert "Simulation Complete!" in events
        assert pathlib.Path(__file__).parent.joinpath("example_data", file_name).read_text() not in events
        assert alert.get_status(run_id) == "ok"
    
    else:
        # FDS failed and has ERRORs in the stderr, but return code is still zero - mark run as failed
        
        assert run_data.status == "failed"
        assert "Simulation Failed!" in events
        assert pathlib.Path(__file__).parent.joinpath("example_data", file_name).read_text() in events
        assert alert.get_status(run_id) == "critical"

    # If slice parsing enabled, check appropriate log message added for no results found
    if slice_parse_enabled:
        assert any(f"Unable to load slice data found in output directory '{tempd}'. Slice parsing is disabled for this run." in event for event in events)