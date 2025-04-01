"""FDS Connector.

This module provides functionality for using Simvue to track and monitor an FDS (Fire Dynamics Simulator) simulation.
"""

import csv
import glob
import pathlib
import platform
import re
import resource
import typing
from datetime import datetime
from itertools import chain

import click
import f90nml
import multiparser.parsing.file as mp_file_parser
import multiparser.parsing.tail as mp_tail_parser
import numpy
import pydantic
import simvue
from simvue.models import DATETIME_FORMAT
from simvue_connector.connector import WrappedRun
from simvue_connector.extras.create_command import format_command_env_vars


class FDSRun(WrappedRun):
    """Class for setting up Simvue tracking and monitoring of an FDS simulation.

    Use this class as a context manager, in the same way you use default Simvue runs, and call run.launch(). Eg:

    with FDSRun() as run:
        run.init(
            name="fds_simulation",
        )
        run.launch(...)
    """

    fds_input_file_path: pydantic.FilePath = None
    workdir_path: typing.Union[str, pydantic.DirectoryPath] = None
    upload_files: typing.List[str] = None
    ulimit: typing.Union[str, int] = None
    fds_env_vars: typing.Dict[str, typing.Any] = None

    _activation_times: bool = False
    _activation_times_data: typing.Dict[str, float] = {}
    _chid: str = None
    _results_prefix: str = None
    _loading_historic_run: bool = False
    _timestamp_mapping: numpy.ndarray = numpy.empty((0, 2))
    _patterns: typing.List[typing.Dict[str, typing.Pattern]] = [
        {"pattern": re.compile(r"\s+Time\sStep\s+(\d+)\s+([\w\s:,]+)"), "name": "step"},
        {
            "pattern": re.compile(r"\s+Step\sSize:.*Total\sTime:\s+([\d\.]+)\ss.*"),
            "name": "time",
        },
        {
            "pattern": re.compile(r"\s+Pressure\sIterations:\s(\d+)$"),
            "name": "pressure_iteration",
        },
        {
            "pattern": re.compile(
                r"\s+Maximum\sVelocity\sError:\s+([\d\.\-\+E]+)\son\sMesh\s\d+\sat\s\(\d+,\d+,\d+\)$"
            ),
            "name": "max_velocity_error",
        },
        {
            "pattern": re.compile(
                r"\s+Maximum\sPressure\sError:\s+([\d\.\-\+E]+)\son\sMesh\s\d+\sat\s\(\d+,\d+,\d+\)$"
            ),
            "name": "max_pressure_error",
        },
        {
            "pattern": re.compile(
                r"\s+Max\sCFL\snumber:\s+([\d\.E\-\+]+)\sat\s\(\d+,\d+,\d+\)$"
            ),
            "name": "max_cfl",
        },
        {
            "pattern": re.compile(
                r"\s+Max\sdivergence:\s+([\d\.E\-\+]+)\sat\s\(\d+,\d+,\d+\)$"
            ),
            "name": "max_divergence",
        },
        {
            "pattern": re.compile(
                r"\s+Min\sdivergence:\s+([\d\.E\-\+]+)\sat\s\(\d+,\d+,\d+\)$"
            ),
            "name": "min_divergence",
        },
        {
            "pattern": re.compile(
                r"\s+Max\sVN\snumber:\s+([\d\.E\-\+]+)\sat\s\(\d+,\d+,\d+\)$"
            ),
            "name": "max_vn",
        },
        {
            "pattern": re.compile(r"\s+No.\sof\sLagrangian\sParticles:\s+(\d+)$"),
            "name": "num_lagrangian_particles",
        },
        {
            "pattern": re.compile(r"\s+Total\sHeat\sRelease\sRate:\s+([\d\.\-]+)\skW$"),
            "name": "total_heat_release_rate",
        },
        {
            "pattern": re.compile(
                r"\s+Radiation\sLoss\sto\sBoundaries:\s+([\d\.\-]+)\skW$"
            ),
            "name": "radiation_loss_to_boundaries",
        },
        {"pattern": re.compile(r"^\s+Mesh\s+(\d+)"), "name": "mesh"},
    ]

    def _soft_abort(self):
        """Create a '.stop' file so that FDS simulation is stopped gracefully if an abort is triggered."""
        if not pathlib.Path(f"{self._results_prefix}.stop").exists():
            with open(f"{self._results_prefix}.stop", "w") as stop_file:
                stop_file.write("FDS simulation aborted due to Simvue Alert.")
                stop_file.close()

    def _estimate_timestamp(self, time_to_convert: float) -> str:
        """If loading historic runs, estimate the timestamp of a data point from a CSV file based on the in-simulation time.

        Since we only have timestamps from the log, and the time steps reported by the log
        won't necessarily correspond to those reported in DEVC / HRR CSV files, we need to
        do some basic interpolation between points in the mapping.

        Parameters
        ----------
        time_to_convert : float
            The time from the CSV file to convert into a timestamp

        Returns
        -------
        str
            The estimated timestamp when this value was recorded

        """
        _index = numpy.searchsorted(self._timestamp_mapping[:, 0], time_to_convert)
        # Find how far between the time values in the mapping the new time is, from to 1
        _fraction = (time_to_convert - self._timestamp_mapping[_index, 0]) / (
            self._timestamp_mapping[_index + 1, 0] - self._timestamp_mapping[_index, 0]
        )
        # Estimate timestamp
        _timestamp = self._timestamp_mapping[_index, 1] + _fraction * (
            self._timestamp_mapping[_index + 1, 1] - self._timestamp_mapping[_index, 1]
        )
        # Convert to string
        return datetime.fromtimestamp(_timestamp).strftime(DATETIME_FORMAT)

    @mp_tail_parser.log_parser
    def _log_parser(
        self, file_content: str, **__
    ) -> tuple[dict[str, typing.Any], list[dict[str, typing.Any]]]:
        """Parse an FDS log file line by line as it is written, and extracts relevant information.

        Parameters
        ----------
        file_content : str
            The next line of the log file
        **__
            Additional unused keyword arguments

        Returns
        -------
        tuple[dict[str, typing.Any], list[dict[str, typing.Any]]]
            An (empty) dictionary of metadata, and a dictionary of metrics data extracted from the log

        """
        _out_data = []
        _out_record = {}
        _current_mesh = None

        # Loop through each line and check against each pattern defined above to extract metrics
        # Remember FDS writes in blocks, so will write at least one full step of data at a time
        for line in file_content.split("\n"):
            for pattern in self._patterns:
                match = pattern["pattern"].search(line)
                if match:
                    # For FDS files with multiple meshes, all metrics will be reported for each mesh
                    # So if this line represents start of a new mesh, update the prefix to be used
                    # for the following metrics to represent this mesh number, then break
                    if pattern["name"] == "mesh":
                        _current_mesh = match.group(1)
                        break

                    # This represents the start of a new block of metric data, so add the previous entry
                    # to the list of all data to return, then start a new entry. Parse the timestamp and
                    # add it as a key to the entry. Dont worry about adding the step itself here, that will
                    # be added in the same way as other metrics below.
                    if pattern["name"] == "step":
                        if _out_record:
                            _out_data += [_out_record]

                        # Get timestamp, get rid of multiple spaces, then format as required
                        _timestamp_str = match.group(2)
                        _timestamp_str = " ".join(_timestamp_str.split())
                        _timestamp = datetime.strptime(
                            _timestamp_str, "%B %d, %Y %H:%M:%S"
                        )

                        # Create new record, reset current mesh to None (to be set later)
                        _out_record = {
                            "timestamp": _timestamp.strftime(DATETIME_FORMAT)
                        }
                        _current_mesh = None

                    # Define the name of the metric, then log the value as key/value pairs
                    # This also adds the 'time' and 'step' for this set of metrics to the dict
                    _metric_name = pattern["name"]
                    if _current_mesh:
                        _metric_name = f"{_metric_name}.mesh.{_current_mesh}"

                    _out_record[_metric_name] = match.group(1)

                    # Log an event for each new step being created
                    if pattern["name"] == "time":
                        self.log_event(
                            f"Time Step: {_out_record['step']}, Simulation Time: {_out_record['time']} s",
                            timestamp=_out_record["timestamp"],
                        )
                        # If loading from historic runs, keep track of time to timestamp mapping
                        if self._loading_historic_run:
                            self._timestamp_mapping = numpy.vstack(
                                (
                                    self._timestamp_mapping,
                                    [
                                        _out_record["time"],
                                        datetime.strptime(
                                            _out_record["timestamp"], DATETIME_FORMAT
                                        ).timestamp(),
                                    ],
                                )
                            )

                    break

            if "DEVICE Activation Times" in line:
                self._activation_times = True
            elif self._activation_times and "Time Stepping" in line:
                self._activation_times = False
            elif self._activation_times:
                match = re.match(r"\s+\d+\s+([\w]+)\s+\w+\s+([\d\.]+)\s*", line)
                if match:
                    self._activation_times_data[f"{match.group(1)}_activation_time"] = (
                        float(match.group(2))
                    )

        # Add the dict of metrics created in the final iterations of the loop
        if _out_record:
            _out_data += [_out_record]

        return {}, _out_data

    def _metrics_callback(self, data: typing.Dict, meta: typing.Dict):
        """Log metrics extracted from a log file to Simvue.

        Parameters
        ----------
        data : typing.Dict
            Dictionary of data to log to Simvue as metrics
        meta: typing.Dict
            Dictionary of metadata added by Multiparser about this data

        """
        metric_time = data.pop("time", None) or data.pop("Time", None)
        metric_step = data.pop("step", None)

        # If this metric is coming from a historic run, we need to estimate the timestamp
        if self._loading_historic_run and not data.get("timestamp"):
            data["timestamp"] = self._estimate_timestamp(metric_time)

        metric_timestamp = data.pop("timestamp", meta["timestamp"].replace(" ", "T"))
        self.log_metrics(
            data, time=metric_time, step=metric_step, timestamp=metric_timestamp
        )

    @mp_file_parser.file_parser
    def _header_metadata(
        self, input_file: str, **__
    ) -> tuple[dict[str, typing.Any], list[dict[str, typing.Any]]]:
        """Parse metadata from header of FDS stderr output file.

        Parameters
        ----------
        input_file : str
            The path to the FDS stderr output file.
        **__
            Additional unused keyword arguments

        Returns
        -------
        tuple[dict[str, typing.Any], list[dict[str, typing.Any]]]
            An (empty) dictionary of metadata, and a dictionary of data to upload as metadata to the Simvue run

        """
        with open(input_file) as in_f:
            _file_lines = in_f.readlines()

        # If loading historic runs, no point looking through entire .out file - just look at first 20 lines:
        if self._loading_historic_run:
            _file_lines = _file_lines[:20]

        _components_regex: dict[str, typing.Pattern[typing.AnyStr]] = {
            "revision": re.compile(r"^\s*Revision\s+\:\s*([\w\d\.\-\_][^\n]+)"),
            "revision_date": re.compile(
                r"^\s*Revision Date\s+\:\s*([\w\s\:\d\-][^\n]+)"
            ),
            "compiler": re.compile(
                r"^\s*Compiler\s+\:\s*([\w\d\-\_\(\)\s\.\[\]\,][^\n]+)"
            ),
            "compilation_date": re.compile(
                r"^\s*Compilation Date\s+\:\s*([\w\d\-\:\,\s][^\n]+)"
            ),
            "mpi_processes": re.compile(r"^\s*Number of MPI Processes:\s*(\d+)"),
            "mpi_version": re.compile(r"^\s*MPI version:\s*([\d\.]+)"),
            "mpi_library_version": re.compile(
                r"^\s*MPI library version:\s*([\w\d\.\s\*\(\)\[\]\-\_][^\n]+)"
            ),
        }

        _output_metadata: dict[str, str] = {"fds": {}}

        for line in _file_lines:
            for key, regex in _components_regex.items():
                if search_res := regex.findall(line):
                    _output_metadata["fds"][key] = search_res[0]

        return {}, _output_metadata

    def _ctrl_log_callback(self, data: typing.Dict, _):
        """Log metrics extracted from the CTRL log file to Simvue.

        Parameters
        ----------
        data : typing.Dict
            Dictionary of data from the latest line of the CTRL log file.

        """
        if data["State"].lower() == "f":
            state = False
        elif data["State"].lower() == "t":
            state = True
        else:
            state = data["State"]

        event_str = f"{data['Type']} '{data['ID']}' has been set to '{state}' at time {data['Time (s)']}s"
        if data.get("Value"):
            event_str += (
                f", when it reached a value of {data['Value']}{data.get('Units', '')}."
            )

        # If loading from historic run, estimate timestamp when this activation was recorded
        # Otherwise, just use the current time
        _timestamp = None
        if self._loading_historic_run:
            _timestamp = self._estimate_timestamp(data.get("Time (s)"))

        self.log_event(event_str, timestamp=_timestamp)
        self.update_metadata({data["ID"]: state})

    def _pre_simulation(self):
        """Start the FDS process."""
        super()._pre_simulation()
        self.log_event("Starting FDS simulation")

        # Save the FDS input file for this run to the Simvue server
        if pathlib.Path(self.fds_input_file_path).exists:
            self.save_file(self.fds_input_file_path, "input")

        # Set stack limit - analogous to 'ulimit -s' recommended in FDS documentation
        if platform.system() != "Windows":
            if self.ulimit == "unlimited" and platform.system() == "Darwin":
                self.log_event(
                    "Warning: Unlimited stack is not supported in MacOS - leaving unchanged."
                )
            elif self.ulimit == "unlimited":
                resource.setrlimit(
                    resource.RLIMIT_STACK,
                    (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
                )
            elif self.ulimit:
                resource.setrlimit(
                    resource.RLIMIT_STACK, (int(self.ulimit), int(self.ulimit))
                )

        def check_for_errors(status_code, std_out, std_err):
            """Need to check for 'ERROR' in logs, since FDS returns rc=0 even if it throws an error."""
            if "ERROR" in std_err:
                click.secho(
                    "[simvue] Run failed - FDS encountered an error: " f"{std_err}",
                    fg="red" if self._term_color else None,
                    bold=self._term_color,
                )
                self._failed = True
                self.log_event("FDS encountered an error:")
                self.log_event(std_err)
                self.log_alert(
                    identifier=self._executor._alert_ids["fds_simulation"],
                    state="critical",
                )
                self.kill_all_processes()
                self._trigger.set()

        command = []
        if self.run_in_parallel:
            command += ["mpiexec", "-n", str(self.num_processors)]
            command += format_command_env_vars(self.mpiexec_env_vars)
        command += ["fds", str(self.fds_input_file_path)]
        command += format_command_env_vars(self.fds_env_vars)

        self.add_process(
            "fds_simulation",
            *command,
            cwd=self.workdir_path,
            completion_trigger=self._trigger,
            completion_callback=check_for_errors,
        )

    def _during_simulation(self):
        """Describe which files should be monitored during the simulation by Multiparser."""
        # Upload data from input file as metadata
        self.file_monitor.track(
            path_glob_exprs=str(self.fds_input_file_path),
            callback=lambda data, meta: self.update_metadata(
                {"input_file": {k: v for k, v in data.items() if v}}
            ),
            file_type="fortran",
            static=True,
        )
        # Upload metadata from file header
        self.file_monitor.track(
            path_glob_exprs=f"{self._results_prefix}.out",
            parser_func=self._header_metadata,
            callback=lambda data, meta: self.update_metadata({**data, **meta}),
            static=True,
        )
        self.file_monitor.tail(
            path_glob_exprs=f"{self._results_prefix}.out",
            parser_func=self._log_parser,
            callback=self._metrics_callback,
        )
        self.file_monitor.tail(
            path_glob_exprs=f"{self._results_prefix}_hrr.csv",
            parser_func=mp_tail_parser.record_csv,
            parser_kwargs={"header_pattern": "Time"},
            callback=self._metrics_callback,
        )
        self.file_monitor.tail(
            path_glob_exprs=f"{self._results_prefix}_devc.csv",
            parser_func=mp_tail_parser.record_csv,
            parser_kwargs={"header_pattern": "Time"},
            callback=self._metrics_callback,
        )
        self.file_monitor.tail(
            path_glob_exprs=f"{self._results_prefix}_devc_ctrl_log.csv",
            parser_func=mp_tail_parser.record_csv,
            callback=self._ctrl_log_callback,
        )

    def _post_simulation(self):
        """Upload files selected by user to Simvue for storage."""
        self.update_metadata(self._activation_times_data)

        if self.upload_files is None:
            for file in glob.glob(f"{self._results_prefix}*"):
                if (
                    pathlib.Path(file).absolute()
                    == pathlib.Path(self.fds_input_file_path).absolute()
                ):
                    continue
                self.save_file(file, "output")
        else:
            if self.workdir_path:
                self.upload_files = [
                    str(pathlib.Path(self.workdir_path).joinpath(path))
                    for path in self.upload_files
                ]

            for path in self.upload_files:
                for file in glob.glob(path):
                    if (
                        pathlib.Path(file).absolute()
                        == pathlib.Path(self.fds_input_file_path).absolute()
                    ):
                        continue
                    self.save_file(file, "output")

        super()._post_simulation()

    @simvue.utilities.prettify_pydantic
    @pydantic.validate_call
    def launch(
        self,
        fds_input_file_path: pydantic.FilePath,
        workdir_path: typing.Union[str, pydantic.DirectoryPath] = None,
        clean_workdir: bool = False,
        upload_files: list[str] = None,
        ulimit: typing.Union[str, int] = "unlimited",
        fds_env_vars: typing.Optional[typing.Dict[str, typing.Any]] = None,
        run_in_parallel: bool = False,
        num_processors: int = 1,
        mpiexec_env_vars: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        """Command to launch the FDS simulation and track it with Simvue.

        Parameters
        ----------
        fds_input_file_path : pydantic.FilePath
            Path to the FDS input file to use in the simulation
        workdir_path : typing.Union[str, pydantic.DirectoryPath], optional
            Path to a directory which you would like FDS to run in, by default None
            This is where FDS will generate the results from the simulation
            If a directory does not already exist at this path, it will be created
            Uses the current working directory by default.
        clean_workdir : bool, optional
            Whether to remove FDS related files from the working directory, by default False
            Useful when doing optimisation problems to remove results from previous runs.
        upload_files : list[str], optional
            List of results file names to upload to the Simvue server for storage, by default None
            These should be supplied as relative to the working directory specified above (if specified, otherwise relative to cwd)
            If not specified, will upload all files by default. If you want no results files to be uploaded, provide an empty list.
        ulimit : typing.Union[str, int], optional
            Value to set your stack size to (for Linux and MacOS), by default "unlimited"
        fds_env_vars : typing.Optional[typing.Dict[str, typing.Any]], optional
            Environment variables to provide to FDS when executed, by default None
        run_in_parallel: bool, optional
            Whether to run the FDS simulation in parallel, by default False
        num_processors : int, optional
            The number of processors to run a parallel FDS job across, by default 1
        mpiexec_env_vars : typing.Optional[typing.Dict[str, typing.Any]]
            Any environment variables to pass to mpiexec on startup if running in parallel, by default None

        """
        self.fds_input_file_path = fds_input_file_path
        self.workdir_path = workdir_path
        self.upload_files = upload_files
        self.ulimit = ulimit
        self.fds_env_vars = fds_env_vars or {}
        self.run_in_parallel = run_in_parallel
        self.num_processors = num_processors
        self.mpiexec_env_vars = mpiexec_env_vars or {}

        self._activation_times = False
        self._activation_times_data = {}

        nml = f90nml.read(self.fds_input_file_path)
        self._chid = nml["head"]["chid"]

        if self.workdir_path:
            pathlib.Path(self.workdir_path).mkdir(exist_ok=True)

            if clean_workdir:
                for file in pathlib.Path(self.workdir_path).glob(f"{self._chid}*"):
                    if (
                        pathlib.Path(file).absolute()
                        == pathlib.Path(self.fds_input_file_path).absolute()
                    ):
                        continue
                    pathlib.Path(file).unlink()

        self._results_prefix = (
            str(pathlib.Path(self.workdir_path).joinpath(self._chid))
            if self.workdir_path
            else self._chid
        )

        super().launch()

    def load(self, results_dir: pydantic.DirectoryPath, upload_files: list[str] = None):
        """Load a pre-existing FDS simulation into Simvue.

        Parameters
        ----------
        results_dir : pydantic.DirectoryPath
            The directory where the results are stored
        upload_files : list[str], optional
            List of results file names to upload to the Simvue server for storage, by default None
            These should be supplied as relative to the results directory specified above
            If not specified, will upload all files by default. If you want no results files to be uploaded, provide an empty list.

        Raises
        ------
        FileNotFoundError
            Raised if no FDS input file was found in specified directory
        ValueError
            Raised if more than one FDS input file found in specified directory

        """
        self.upload_files = upload_files
        self._loading_historic_run = True

        # Find input file inside results dir
        _fds_files = list(pathlib.Path(results_dir).rglob("*.fds"))

        if not _fds_files:
            raise FileNotFoundError(
                "Could not find an input '.fds' file in your results directory. Please make sure the relevant input file is provided."
            )
        elif len(_fds_files) > 1:
            raise ValueError(
                "Found more than one input '.fds' file - please make sure only one such file is in the results directory."
            )

        self.fds_input_file_path = _fds_files[0]
        self.save_file(self.fds_input_file_path, "input")

        # Load input file, upload as metadata
        _nml = f90nml.read(self.fds_input_file_path)
        self._chid = _nml["head"]["chid"]
        self.update_metadata({"input_file": _nml})

        self._results_prefix = str(pathlib.Path(self.results_dir).joinpath(self._chid))

        # TODO: Get start time from head of log, end time from last timestamp in log, upload to run (how? Direct from Run api object?)

        # Read relevant files and call methods directly

        # Extract metadata from log file header
        _data, _meta = self._header_metadata(f"{self._results_prefix}.out")
        self.update_metadata({**_data, **_meta})

        # Extract metrics from log file
        with open(f"{self._results_prefix}.out", "r") as log_file:
            _, _log_metrics = self._log_parser(log_file)
        for _metric in _log_metrics:
            self._metrics_callback(_metric)

        # Extract metrics from DEVC and HRR files
        with open(f"{self._results_prefix}_hrr.csv", "r") as hrr_file:
            # Skip the first line, as that contains units and not the header names we want
            next(hrr_file)
            _hrr_metrics = csv.DictReader(hrr_file)

        with open(f"{self._results_prefix}_devc.csv", "r") as devc_file:
            # Skip the first line, as that contains units and not the header names we want
            next(devc_file)
            _devc_metrics = csv.DictReader(devc_file)

        for _metric in chain(_hrr_metrics, _devc_metrics):
            self._metrics_callback(_metric)

        # Extract events / metadata from CTRL log file
        with open(f"{self._results_prefix}_devc_ctrl_log.csv", "r") as ctrl_file:
            _ctrl_data = csv.DictReader(devc_file)

        for _metric in _ctrl_data:
            self._ctrl_log_callback(_metric)

        self._post_simulation()
