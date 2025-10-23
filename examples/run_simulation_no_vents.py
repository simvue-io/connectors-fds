"""
FDS Connector Example
========================
This is an example of the FDSRun Connector class.

The FDS simulation used here simulates a fire starting in the centre of a small room,
without any mitigation

To run this example with Docker:
    - Pull the FDS image: docker run -it ghcr.io/simvue-io/fds_example
    - Create a simvue.toml file, copying in your information from the Simvue server: nano simvue.toml
    - Run the example script: python run_simulation_no_vents.py

To run this example on your own system with FDS installed:
    - Ensure that you have FDS installed and added to your path: `fds` on UNIX or `fds_local` on Windows
    - Clone this repository: git clone https://github.com/simvue-io/connectors-fds.git
    - Move into FDS examples directory: cd connectors-fds/examples/fds
    - Create a simvue.toml file, copying in your information from the Simvue server: vi simvue.toml
    - Create a venv: python -m venv venv
    - Activate the venv: `source venv/bin/activate` on UNIX, or `venv/Scripts/activate` on Windows
    - Install required modules: pip install .
    - Run the example script: poetry run python run_simulation_no_vents.py
    
For a more in depth example, see: https://docs.simvue.io/examples/fds/
"""
import pathlib
from simvue_fds.connector import FDSRun

# Use the FDSRun class as a context manager
with FDSRun() as run:
    # Initialize the run with a name, and optional other parameters
    run.init(
        name="fds_simulation_no_vents", 
        folder="/examples/fds")

    # Call any other simvue Run() methods, such as adding tags or alerts
    run.update_tags(["fds", "no_vents"])

    run.create_metric_threshold_alert(
        name="visibility_below_three_metres",
        metric="eye_level_visibility",
        frequency=1,
        window=1,
        rule="is below",
        threshold=3,
        trigger_abort=True,
    )
    
    run.create_metric_threshold_alert(
        name="average_temperature_above_100_degrees",
        metric="temperature.y.2_0.avg",
        frequency=1,
        window=1,
        rule="is above",
        threshold=100,
        trigger_abort=True,
    )

    # Launch the FDS run, providing the path to your input file
    run.launch(
        fds_input_file_path=pathlib.Path(__file__).parent.joinpath("input_no_vents.fds"),
        workdir_path=str(pathlib.Path(__file__).parent.joinpath("results_no_vents")),
        clean_workdir=True,
        slice_parse_quantity = "TEMPERATURE",
    )
        