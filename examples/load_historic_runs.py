"""
Loading FDS Runs Example
==========================
This is an example of the FDSRun Connector class being used to load historic FDS simulation data.

The FDS simulation used here simulates a fire starting in the centre of a small room,
with supply and exhaust air vents activating after a few seconds to remove the smoke.

This example does not require installation of FDS, as the results have already been generated.
View the results in the folder `examples/example_results`.

To run this example:
    - Clone this repository: git clone https://github.com/simvue-io/connectors-fds.git
    - Move into FDS examples directory: cd connectors-fds/examples
    - Create a simvue.toml file, copying in your information from the Simvue server: vi simvue.toml
    - Install Poetry: pip install poetry
    - Install required modules: poetry install
    - Run the example script: poetry run python load_historic_runs.py
    
For a more in depth example, see: https://docs.simvue.io/examples/fds/
"""

import pathlib
import shutil
import uuid
from simvue_fds.connector import FDSRun


# Initialise the FDSRun class as a context manager
with FDSRun() as run:
    # Initialise the run, providing a name for the run, and optionally extra information such as a folder, description, tags etc
    run.init(
        name="fds_simulation_vents-%s" % str(uuid.uuid4()),
        description="An example of using the FDSRun Connector to load an FDS simulation.",
        folder="/simvue_client_examples/fds",
        tags=["fds", "vents"],
    )

    # Call the .load() method to load historic results, providing the path to the results directory
    run.load(
        results_dir = pathlib.Path(__file__).parent.joinpath("example_results"),
        # You can optionally have the connector track slices in your simulation
        slice_parse_quantity = "TEMPERATURE",
    )
    
    # Once the simulation is complete, you can upload any final items to the Simvue run before it closes
    run.log_event("Finished uploading results stored in directory 'example_results'!")

