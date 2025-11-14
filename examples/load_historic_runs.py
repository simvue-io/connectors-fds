"""
Loading FDS Runs Example
==========================
This is an example of the FDSRun Connector class being used to load historic FDS simulation data.

The FDS simulation used here simulates a fire starting in the centre of a small room,
with supply and exhaust air vents activating after a few seconds to remove the smoke.

This example does not require installation of FDS, as the results have already been generated.
View the results in the folder `examples/example_results`.

To run this example with Docker:
    - Pull the FDS image: docker run -it ghcr.io/simvue-io/fds_example
    - Create a simvue.toml file, copying in your information from the Simvue server: nano simvue.toml
    - Run the example script: python load_historic_runs.py

To run this example on your own system:
    - Clone this repository: git clone https://github.com/simvue-io/connectors-fds.git
    - Move into FDS examples directory: cd connectors-fds/examples/fds
    - Create a simvue.toml file, copying in your information from the Simvue server: vi simvue.toml
    - Create a venv: python -m venv venv
    - Activate the venv: `source venv/bin/activate` on UNIX, or `venv/Scripts/activate` on Windows
    - Install required modules: pip install .
    - Run the example script: poetry run python load_historic_results.py
    
For a more in depth example, see: https://docs.simvue.io/examples/fds/
"""
import pathlib
from simvue_fds.connector import FDSRun

# Use the FDSRun class as a context manager
with FDSRun() as run:
    # Initialize the run with a name, and optional other parameters
    run.init(
        name="fds_simulation_stronger_vents",
        folder="/examples/fds"
        )

    # Call any other simvue Run() methods, such as adding tags or alerts
    run.update_tags(["fds", "vents"])

    # Call run.load(), providing the path to your directory of results
    run.load(
        results_dir=pathlib.Path(__file__).parent.joinpath("example_results"),
        slice_parse_quantity = "TEMPERATURE"
    )