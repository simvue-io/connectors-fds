"""
FDS Connector Example
========================
This is an example of the FDSRun Connector class.

The FDS simulation used here simulates a fire starting in the centre of a small room,
with supply and exhaust air vents activating after a few seconds to remove the smoke.

To run this example with Docker:
    - Pull the base FDS image: docker run -it ghcr.io/simvue-io/fds_example
    - Clone this repository: git clone https://github.com/simvue-io/connectors-fds.git
    - Move into FDS examples directory: cd connectors-fds/examples
    - Create a simvue.toml file, copying in your information from the Simvue server: nano simvue.toml
    - Install Poetry: pip install poetry
    - Install required modules: poetry install
    - Run the example script: poetry run python fds_example.py

To run this example on your own system with FDS installed:
    - Ensure that you have FDS installed and added to your path: fds --help
    - Clone this repository: git clone https://github.com/simvue-io/connectors-fds.git
    - Move into FDS examples directory: cd connectors-fds/examples/fds
    - Create a simvue.toml file, copying in your information from the Simvue server: vi simvue.toml
    - Install Poetry: pip install poetry
    - Install required modules: poetry install
    - Run the example script: poetry run python fds_example.py
    
For a more in depth example, see: https://docs.simvue.io/examples/fds/
"""

import pathlib
import shutil
import uuid
from simvue_fds.connector import FDSRun

def fds_example(run_folder, offline=False, parallel=False) -> None:
    
    # Delete old copies of results, if they exist:
    if pathlib.Path(__file__).parent.joinpath("results").exists():
        shutil.rmtree(pathlib.Path(__file__).parent.joinpath("results"))

    # Initialise the FDSRun class as a context manager
    with FDSRun(mode="offline" if offline else "online") as run:
        # Initialise the run, providing a name for the run, and optionally extra information such as a folder, description, tags etc
        run.init(
            name="fds_simulation_vents-%s" % str(uuid.uuid4()),
            description="An example of using the FDSRun Connector to track an FDS simulation.",
            folder=run_folder,
            tags=["fds", "vents"],
        )
        
        # You can use any of the Simvue Run() methods to upload extra information before/after the simulation
        run.create_metric_threshold_alert(
            name="visibility_below_three_metres",
            metric="eye_level_visibility",
            frequency=1,
            rule="is below",
            threshold=3,
        )

        # Then call the .launch() method to start your FDS simulation, providing the path to the input file
        run.launch(
            fds_input_file_path = pathlib.Path(__file__).parent.joinpath("activate_vents.fds"),
            workdir_path = str(pathlib.Path(__file__).parent.joinpath("results")),
            # And you can choose whether to run it in parallel
            run_in_parallel = parallel,
            num_processors = 2,
        )
        
        # Once the simulation is complete, you can upload any final items to the Simvue run before it closes
        run.log_event("Deleting local copies of results...")
        
        return run.id

if __name__ == "__main__":
    fds_example("/fds_example")

