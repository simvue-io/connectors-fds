import os
import shutil
import pytest
import pathlib
import platform
import tempfile
import numpy
import simvue
import time
from simvue.sender import sender
from simvue_fds.connector import FDSRun
import uuid

@pytest.fixture
def find_bin():
    if platform.system() != "Windows":
        # Find path to FDS executable
        fds_bin = shutil.which("fds")
    else:
        search_paths = [
            pathlib.Path(os.environ["PROGRAMFILES"]).joinpath("firemodels"),
            pathlib.Path(os.environ["LOCALAPPDATA"]).joinpath("firemodels"),
            pathlib.Path.home().joinpath("firemodels"),
        ]
        if os.environ.get("GITHUB_WORKSPACE"):
            search_paths.append(pathlib.Path(os.environ["GITHUB_WORKSPACE"]).joinpath("firemodels"))

        for search_loc in search_paths:
            if not search_loc.exists():
                continue
            if search := pathlib.Path(search_loc).rglob("**/fds_local.bat"):
                fds_bin = f"{next(search)}"
                break
            
    if not fds_bin:
        raise pytest.skip("FDS executable could not be found!")
    
    return fds_bin

def run_fds(file_path, run_folder, parallel, offline, slice_var, load):
    """Function demonstrating how to launch FDS runs with Simvue.

    Parameters
    ----------
    file_path : str
        Path to FDS input file
    run_folder : str
        The folder/directory where the input file is stored
    offline : bool, optional
        Whether to run in offline mode, by default False
    parallel : bool, optional
        Whether to run FDS across multiple CPU cores in parallel, by default False
    Returns
    -------
    str
        The ID of the run
    """
    # Delete old copies of results, if they exist:
    if pathlib.Path(__file__).parent.joinpath("results").exists():
        shutil.rmtree(pathlib.Path(__file__).parent.joinpath("results"))

    # Initialise the FDSRun class as a context manager
    with FDSRun(mode="offline" if offline else "online") as run:
        # Initialise the run, providing a name for the run, and optionally extra information such as a folder, description, tags etc
        run.init(
            name=f"fds-integration-{file_path.stem}-{'parallel' if parallel else 'serial'}-{'offline' if offline else 'online'}-{'load' if load else 'launch'}-{str(uuid.uuid4())}",
            description="An example of using the FDSRun Connector to track an FDS simulation.",
            folder=run_folder,
            tags=["fds", "integration", "test"],
        )
        # You can use any of the Simvue Run() methods to upload extra information before/after the simulation
        run.create_metric_threshold_alert(
            name="avg_temp_above_100",
            metric="temperature.y.2_0.avg",
            frequency=1,
            rule="is above",
            threshold=100,
        )
        if load:
            run.load(
                results_dir=file_path,
                slice_parse_quantity = slice_var,
            )
        else:
            # Then call the .launch() method to start your FDS simulation, providing the path to the input file
            run.launch(
                fds_input_file_path = file_path,
                workdir_path = str(pathlib.Path(__file__).parent.joinpath("results")),
                clean_workdir=True,
                # You can optionally have the connector track slices in your simulation
                slice_parse_quantity = slice_var,
                slice_parse_interval = 10,
                # And you can choose whether to run it in parallel
                run_in_parallel = parallel,
                num_processors = 2,
            )
        
        # Once the simulation is complete, you can upload any final items to the Simvue run before it closes
        run.log_event("Deleting local copies of results...")
        
        run_id = run.id
        
        if offline:
            _id_mapping = sender(throw_exceptions=True)
            run_id = _id_mapping.get(run_id)
        
        return run_id

@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))
@pytest.mark.parametrize("parallel", (True, False), ids=("parallel", "serial"))
@pytest.mark.parametrize("offline", (True, False), ids=("offline", "online"))
def test_fds_launch_supply_exhaust(folder_setup, offline_cache_setup, load, offline, parallel):
    if load:
        if parallel:
            pytest.skip("Parallel has no effect when loading from historic runs")
        file_path = pathlib.Path(__file__).parent.joinpath("example_data", "load", "supply_exhaust_vents")
    else:
        file_path = pathlib.Path(__file__).parent.joinpath("example_data", "launch", "supply_exhaust_vents.fds")

    run_id = run_fds(
        file_path=file_path, 
        run_folder=folder_setup, 
        parallel=parallel, 
        offline=offline, 
        slice_var='TEMPERATURE',
        load=load
        )

    time.sleep(2)

    client = simvue.Client()
    run_data = client.get_run(run_id)
    events = [event["message"] for event in client.get_events(run_id)]

    # Check run description and tags from init have been added
    assert (
        run_data.description
        == "An example of using the FDSRun Connector to track an FDS simulation."
    )
    assert sorted(run_data.tags) == ["fds", "integration", "test"]

    # Check alert has been added
    assert "avg_temp_above_100" in [
        alert["name"] for alert in run_data.get_alert_details()
    ]

    # Check metadata from header
    if parallel:
        assert run_data.metadata["fds"]["mpi_processes"] == "2"
    else:
        assert run_data.metadata["fds"]["mpi_processes"] == "1"

    # Check metadata from input file
    assert run_data.metadata["input_file"]["_grp_devc_1"]["id"] == "flow_volume_supply"

    # Check events from log
    # Loosening requirement for this since Windows and Ubuntu will print slightly different times
    assert any([event.startswith("Time Step: 1, Simulation Time: 0.092") for event in events])

    # Check events from DEVC/CTRL log
    # Loosening requirement for this since Windows and Ubuntu will print slightly different times
    assert any([event.startswith("DEVC 'timer' has been set to 'True' at time 2.") for event in events])

    metrics = dict(run_data.metrics)
    # Check metrics from HRR file
    assert metrics["HRR"]["count"] > 0

    # Check metrics from DEVC file
    assert metrics["flow_volume_supply"]["count"] > 0

    # Check metrics from log file
    assert metrics["max_pressure_error"]["count"] > 0
    assert metrics["max_divergence.mesh.2"]["count"] > 0

    # Check metrics from slice
    assert metrics["temperature.y.2_0.min"]["count"] > 0
    assert metrics["temperature.y.2_0.min"]["count"] > 0
    assert metrics["temperature.y.2_0.min"]["count"] > 0

    _retrieved = client.get_metric_values(
        run_ids=[run_id],
        metric_names=[
            "temperature.y.2_0.max",
            "temperature.y.2_0.min",
            "temperature.y.2_0.avg",
        ],
        xaxis="time",
    )
    _max = numpy.array(list(_retrieved["temperature.y.2_0.max"].values()))
    _min = numpy.array(list(_retrieved["temperature.y.2_0.min"].values()))
    _avg = numpy.array(list(_retrieved["temperature.y.2_0.avg"].values()))

    # Check all max >= avg >= min
    assert numpy.all(_max >= _avg)
    assert numpy.all(_avg >= _min)
    assert numpy.all(_min > 0)

    temp_dir = tempfile.TemporaryDirectory()

    # Check input file uploaded as input
    client.get_artifacts_as_files(run_id, "input", temp_dir.name)
    assert pathlib.Path(temp_dir.name).joinpath("supply_exhaust_vents.fds").exists()

    # Check results uploaded as output
    client.get_artifacts_as_files(run_id, "output", temp_dir.name)
    assert pathlib.Path(temp_dir.name).joinpath("supply_exhaust_vents.smv").exists()


@pytest.mark.parametrize("load", (True, False), ids=("load", "launch"))
@pytest.mark.parametrize("parallel", (True, False), ids=("parallel", "serial"))
@pytest.mark.parametrize("offline", (True, False), ids=("offline", "online"))
def test_fds_launch_aalto_woods(folder_setup, offline_cache_setup, offline, parallel, load):
    if load:
        if parallel:
            pytest.skip("Parallel has no effect when loading from historic runs")
        file_path = pathlib.Path(__file__).parent.joinpath("example_data", "load", "aalto_woods_spruce")
    else:
        file_path = pathlib.Path(__file__).parent.joinpath("example_data", "launch", "aalto_woods_spruce", "spruce_N2_50.fds")

    run_id = run_fds(
        file_path=file_path, 
        run_folder=folder_setup, 
        parallel=parallel, 
        offline=offline, 
        slice_var=None,
        load=load
        )
    time.sleep(2)

    client = simvue.Client()
    run_data = client.get_run(run_id)
    events = [event["message"] for event in client.get_events(run_id)]

    # Check run description and tags from init have been added
    assert (
        run_data.description
        == "An example of using the FDSRun Connector to track an FDS simulation."
    )
    assert sorted(run_data.tags) == ["fds", "integration", "test"]

    # Check alert has been added
    assert "avg_temp_above_100" in [
        alert["name"] for alert in run_data.get_alert_details()
    ]
    
    # Check metadata from header
    if parallel:
        assert run_data.metadata["fds"]["mpi_processes"] == "2"
    else:
        assert run_data.metadata["fds"]["mpi_processes"] == "1"

    # Check metadata from input file
    assert run_data.metadata["input_file"]["time"]["t_end"] == 1750
    
    # Check metadata from concatenated files
    assert run_data.metadata["input_file"]["_grp_devc_0"]["id"] == "HRRPUA"
    assert run_data.metadata["input_file"]["_grp_spec_0"]["id"] == "WATER VAPOR"
    assert run_data.metadata["input_file"]["reac"]["fuel"] == "PYROLYZATE"

    # Check events from log
    assert any([event.startswith("Time Step: 1, Simulation Time: 0.5") for event in events])

    metrics = dict(run_data.metrics)
    # Check metrics from HRR file
    assert metrics["HRR"]["count"] > 0

    # Check metrics from DEVC file
    assert metrics["HRRPUA"]["count"] > 0

    # Check metrics from log file
    assert metrics["max_pressure_error"]["count"] > 0
    assert metrics["max_divergence.mesh.2"]["count"] > 0

    temp_dir = tempfile.TemporaryDirectory()

    # Check CONCATENATED input file uploaded as input
    client.get_artifacts_as_files(run_id, "input", temp_dir.name)
    assert pathlib.Path(temp_dir.name).joinpath("spruce_N2_50_cat.fds").exists()

    # Check results uploaded as output
    client.get_artifacts_as_files(run_id, "output", temp_dir.name)
    assert pathlib.Path(temp_dir.name).joinpath("spruce_N2_50_cat.smv").exists()