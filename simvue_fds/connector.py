"""FDS Connector.

This module provides functionality for using Simvue to track and monitor an FDS (Fire Dynamics Simulator) simulation.
"""

import contextlib
import csv
import glob
import io
import os
import pathlib
import platform
import re
import resource
import shutil
import threading
import time
import typing
from datetime import datetime
from itertools import chain

import click
import f90nml
import multiparser.parsing.file as mp_file_parser
import multiparser.parsing.tail as mp_tail_parser
import numpy
import pydantic
import pyfdstools
import simvue
from loguru import logger
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
    slice_parse_quantity: str | None = None
    slice_parse_interval: int = 1
    slice_parse_ignore_zeros: bool = False
    ulimit: typing.Union[str, int] = None
    fds_env_vars: typing.Dict[str, typing.Any] = None

    # Users can set this before launching a simulation, if they want (not in launch to not bloat arguments required)
    upload_input_file: bool = True

    _slice_processed_time: int = -1
    _slice_step: int = 0
    _step_tracker: dict = {}
    _parsing: bool = False
    _parse_time: float = datetime.now().timestamp()
    _activation_times: bool = False
    _activation_times_data: typing.Dict[str, float] = {}
    _chid: str = ""
    _results_prefix: str = ""
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

    def _soft_abort(self) -> None:
        """Create a '.stop' file so that FDS simulation is stopped gracefully if an abort is triggered."""
        if (
            self._results_prefix
            and not (
                _file_path := pathlib.Path(f"{self._results_prefix}.stop")
            ).exists()
        ):
            with _file_path.open("w") as stop_file:
                stop_file.write("FDS simulation aborted due to Simvue Alert.")

    def _tidy_run(self) -> None:
        """Add correct start and end times.

        Override base Run class tidy method to set start and end times according to when simulation was actually ran,
        if loading from historic data.
        """
        super()._tidy_run()
        if self._loading_historic_run and self._timestamp_mapping.size:
            self._sv_obj.started = datetime.fromtimestamp(self._timestamp_mapping[0, 1])
            self._sv_obj.endtime = datetime.fromtimestamp(
                self._timestamp_mapping[-1, 1]
            )
            self._sv_obj.commit()

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

        # If the index found is 0, then it has found a time before our first measurement from out file
        # Just use first measurement from out file as this is the best guess we can achieve
        if _index == 0:
            _timestamp = self._timestamp_mapping[0, 1]
        # If the index found is the last in the array, just use final timestep
        elif _index == self._timestamp_mapping.shape[0]:
            _timestamp = self._timestamp_mapping[-1, 1]
        else:
            # Find how far between the time values in the mapping the new time is, from 0 to 1
            _fraction = (time_to_convert - self._timestamp_mapping[_index - 1, 0]) / (
                self._timestamp_mapping[_index, 0]
                - self._timestamp_mapping[_index - 1, 0]
            )
            # Estimate timestamp
            _timestamp = self._timestamp_mapping[_index - 1, 1] + _fraction * (
                self._timestamp_mapping[_index, 1]
                - self._timestamp_mapping[_index - 1, 1]
            )
        # Convert to string
        return datetime.fromtimestamp(_timestamp).strftime(DATETIME_FORMAT)

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
                                        float(_out_record["time"]),
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
        metric_step = data.pop("step", None) or self._step_tracker.get(
            meta["file_name"], 0
        )
        # If this metric is coming from a historic run, we need to estimate the timestamp
        if self._loading_historic_run:
            metric_timestamp = data.pop(
                "timestamp",
                self._estimate_timestamp(float(metric_time)) if metric_time else None,
            )
        else:
            metric_timestamp = data.pop(
                "timestamp", meta["timestamp"].replace(" ", "T")
            )

        self.log_metrics(
            data, time=metric_time, step=metric_step, timestamp=metric_timestamp
        )
        # Since we don't have 'step' information from CSV files, just increment on each reading, starting from 0
        self._step_tracker[meta["file_name"]] = int(metric_step) + 1

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

    def _ctrl_log_callback(self, data: typing.Dict, *_):
        """Log metrics extracted from the CTRL log file to Simvue.

        Parameters
        ----------
        data : typing.Dict
            Dictionary of data from the latest line of the CTRL log file.
        *_
            Additional unused arguments

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
        _timestamp: str | None = None
        if self._loading_historic_run:
            _timestamp = self._estimate_timestamp(float(data.get("Time (s)")))

        self.log_event(event_str, timestamp=_timestamp)
        self.update_metadata({data["ID"]: state})

    def _add_slice_metrics(
        self,
        metrics: dict,
        sub_slice: numpy.ndarray,
        label: str,
        name: float,
        ignore_zeros: bool,
    ) -> dict:
        """Add metrics for a given slice at a specific time to a dictionary of metrics.

        Parameters
        ----------
        metrics : dict
            The dictionary of metrics to add to
        sub_slice : numpy.ndarray
            The slice to compute min, max, mean over
        label : str
            The dimension which this slice is calculated over
        name : float
            Position in space where the slice is calculated in the above dimension
        ignore_zeros : bool
            Whether to ignore zeros in the slices

        Returns
        -------
        dict
            The updated metrics

        """
        sub_slice = sub_slice[~numpy.isnan(sub_slice)]
        if ignore_zeros:
            sub_slice = sub_slice[numpy.where(sub_slice != 0)]
        _metric_label = f"{self.slice_parse_quantity.replace(' ', '_').lower()}.{label}.{str(round(name, 3)).replace('.', '_')}"
        metrics[f"{_metric_label}.min"] = numpy.min(sub_slice)
        metrics[f"{_metric_label}.max"] = numpy.max(sub_slice)
        metrics[f"{_metric_label}.avg"] = numpy.mean(sub_slice)
        return metrics

    def _parse_slice(self) -> bool:
        """Parse slices present in the FDS results files and extract data as metrics.

        Returns
        -------
        bool
            Whether the slice was successfuly extracted

        """
        # grid_abs is an array of all possible grid points, shape (X, Y, Z, 3)
        # data_abs is an array of all values, shape (X, Y, Z, times)
        # times_out is an array of in simulation times
        temp_stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(temp_stdout):
                grid_abs, data_abs, times_out = pyfdstools.readSLCF2Ddata(
                    self._chid,
                    str(pathlib.Path(self.workdir_path).absolute())
                    if self.workdir_path
                    else str(pathlib.Path.cwd()),
                    self.slice_parse_quantity,
                )
        except Exception as e:
            logger.error(
                """Failed to collect 2D slice data - check that your slice quantity is valid.
                Slice parsing is disabled for this run. Enable debug logging for more info."""
            )
            logger.debug(f"Exception is: {e}")
            logger.debug(f"Debug: \n {temp_stdout.getvalue()}")
            return False

        # Remove times which we have already processed
        times_out = times_out[: data_abs.shape[-1]]
        to_process = numpy.where(times_out > self._slice_processed_time)[0]

        if len(to_process) == 0:
            return True

        times_out = times_out[to_process]
        data_abs = data_abs[:, :, :, to_process]

        # Find X, Y, and Z slices which are present in our data
        # Defined as over 50% of mesh points being not NaN
        x_indices = numpy.where(
            numpy.sum(~numpy.isnan(data_abs[..., 0]), axis=(1, 2))
            / (data_abs.shape[1] * data_abs.shape[2])
            > 0.5
        )[0]
        y_indices = numpy.where(
            numpy.sum(~numpy.isnan(data_abs[..., 0]), axis=(0, 2))
            / (data_abs.shape[0] * data_abs.shape[2])
            > 0.5
        )[0]
        z_indices = numpy.where(
            numpy.sum(~numpy.isnan(data_abs[..., 0]), axis=(0, 1))
            / (data_abs.shape[0] * data_abs.shape[1])
            > 0.5
        )[0]

        # Convert these to their actual positions, for naming
        x_names = grid_abs[x_indices, 0, 0, 0]
        y_names = grid_abs[0, y_indices, 0, 1]
        z_names = grid_abs[0, 0, z_indices, 2]

        # Take the 2D slices
        x_slices = data_abs[x_indices, :, :, :]
        y_slices = data_abs[:, y_indices, :, :]
        z_slices = data_abs[:, :, z_indices, :]

        for time_idx, time_val in enumerate(times_out):
            metrics = {}
            for idx in range(len(x_indices)):
                sub_slice = x_slices[idx, :, :, time_idx]
                metrics = self._add_slice_metrics(
                    metrics,
                    sub_slice,
                    label="x",
                    name=x_names[idx],
                    ignore_zeros=self.slice_parse_ignore_zeros,
                )

            for idx in range(len(y_indices)):
                sub_slice = y_slices[:, idx, :, time_idx]
                metrics = self._add_slice_metrics(
                    metrics,
                    sub_slice,
                    label="y",
                    name=y_names[idx],
                    ignore_zeros=self.slice_parse_ignore_zeros,
                )

            for idx in range(len(z_indices)):
                sub_slice = z_slices[:, :, idx, time_idx]
                metrics = self._add_slice_metrics(
                    metrics,
                    sub_slice,
                    label="z",
                    name=z_names[idx],
                    ignore_zeros=self.slice_parse_ignore_zeros,
                )

            # Need to estimate timestamp which this measurement would correspond to
            # Will use estimate = timestamp of last parse + (now - last parse) * (idx/len(times_out))
            _timestamp_estimate: float = self._parse_time + (
                datetime.now().timestamp() - self._parse_time
            ) * ((time_idx + 1) / len(times_out))

            self.log_metrics(
                metrics,
                time=float(time_val),
                step=self._slice_step,
                timestamp=datetime.fromtimestamp(_timestamp_estimate).strftime(
                    DATETIME_FORMAT
                ),
            )
            self._slice_step += 1
        self._parse_time = datetime.now().timestamp()
        self._slice_processed_time = times_out[-1]
        return True

    def _slice_parser(self) -> None:
        """Read and process all 2D slice files in a loop, uploading min, max and mean as metrics."""
        self._parsing = True
        while True:
            time.sleep(60 * self.slice_parse_interval)

            slice_parsed = self._parse_slice()

            if self._trigger.is_set() or not slice_parsed:
                self._parsing = False
                break

    def _pre_simulation(self):
        """Start the FDS process."""
        super()._pre_simulation()
        self.log_event("Starting FDS simulation")

        # Save the FDS input file for this run to the Simvue server
        if self.upload_input_file and pathlib.Path(self.fds_input_file_path).exists:
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

        if self.slice_parse_quantity:
            slice_parser = threading.Thread(
                target=self._slice_parser, daemon=True, name="slice_parser"
            )
            slice_parser.start()

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
            parser_func=mp_file_parser.file_parser(self._header_metadata),
            callback=lambda data, meta: self.update_metadata({**data, **meta}),
            static=True,
        )
        self.file_monitor.tail(
            path_glob_exprs=f"{self._results_prefix}.out",
            parser_func=mp_tail_parser.log_parser(self._log_parser),
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

        # Then wait for slice parser to finish
        while self._parsing:
            time.sleep(10)

        super()._post_simulation()

    @simvue.utilities.prettify_pydantic
    @pydantic.validate_call
    def launch(
        self,
        fds_input_file_path: pydantic.FilePath,
        workdir_path: typing.Union[str, pydantic.DirectoryPath] = None,
        clean_workdir: bool = False,
        upload_files: list[str] | None = None,
        slice_parse_quantity: str | None = None,
        slice_parse_interval: int = 1,
        slice_parse_ignore_zeros: bool = True,
        ulimit: typing.Literal["unlimited"] | int = "unlimited",
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
        upload_files : list[str] | None, optional
            List of results file names to upload to the Simvue server for storage, by default None
            These should be supplied as relative to the working directory specified above (if specified, otherwise relative to cwd)
            If not specified, will upload all files by default. If you want no results files to be uploaded, provide an empty list.
        slice_parse_quantity: str | None, optional
            ***** WARNING: EXPERIMENTAL FEATURE*****
            The quantity for which to find any 2D slices saved by the simulation, and upload the min/max/average as metrics
            Default is None, which will disable this feature
        slice_parse_interval : int, optional
            Interval (in minutes) at which to parse and upload 2D slice data, default is 1
        slice_parse_ignore_zeros : bool, optional
            Whether to ignore values of zero in slices (useful if there are obstructions in the mesh), default is True
        ulimit : typing.Literal["unlimited"] | int, optional
            Value to set your stack size to (for Linux and MacOS), by default "unlimited"
        fds_env_vars : typing.Optional[typing.Dict[str, typing.Any]], optional
            Environment variables to provide to FDS when executed, by default None
        run_in_parallel: bool, optional
            Whether to run the FDS simulation in parallel, by default False
        num_processors : int, optional
            The number of processors to run a parallel FDS job across, by default 1
        mpiexec_env_vars : typing.Optional[typing.Dict[str, typing.Any]]
            Any environment variables to pass to mpiexec on startup if running in parallel, by default None

        Raises
        ------
        ValueError
            Raised if 2D slices could not be parsed correctly

        """
        self.fds_input_file_path = fds_input_file_path
        self.workdir_path = workdir_path
        self.upload_files = upload_files
        self.slice_parse_quantity = slice_parse_quantity
        self.slice_parse_interval = slice_parse_interval
        self.slice_parse_ignore_zeros = slice_parse_ignore_zeros
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

        if self.slice_parse_quantity:
            # This is only necessary because of the way pyfdstools works
            if (
                self.workdir_path
                and (
                    pathlib.Path(self.fds_input_file_path).absolute()
                    != pathlib.Path(self.workdir_path)
                    .joinpath(f"{self._chid}.fds")
                    .absolute()
                )
            ) or (
                not self.workdir_path
                and (
                    pathlib.Path(self.fds_input_file_path).absolute()
                    != pathlib.Path.cwd().joinpath(f"{self._chid}.fds").absolute()
                )
            ):
                shutil.copy(self.fds_input_file_path, f"{self._results_prefix}.fds")

            # Make sure xyz is enabled
            if not nml.get("dump", {}).get("write_xyz"):
                raise ValueError(
                    "WRITE_XYZ must be enabled in your FDS file for slice parsing."
                )

        super().launch()

    def load(
        self,
        results_dir: pydantic.DirectoryPath,
        upload_files: list[str] | None = None,
        slice_parse_quantity: str | None = None,
        slice_parse_ignore_zeros: bool = True,
    ) -> None:
        """Load a pre-existing FDS simulation into Simvue.

        Parameters
        ----------
        results_dir : pydantic.DirectoryPath
            The directory where the results are stored
        upload_files : list[str] | None, optional
            List of results file names to upload to the Simvue server for storage, by default None
            These should be supplied as relative to the results directory specified above
            If not specified, will upload all files by default. If you want no results files to be uploaded, provide an empty list.
        slice_parse_quantity: str | None, optional
            ***** WARNING: EXPERIMENTAL FEATURE*****
            The quantity for which to find any 2D slices saved by the simulation, and upload the min/max/average as metrics
            Default is None, which will disable this feature
            Note that the XYZ files must have been recorded when running the simulation for this to work.
        slice_parse_ignore_zeros : bool, optional
            Whether to ignore values of zero in slices (useful if there are obstructions in the mesh), default is True

        Raises
        ------
        ValueError
            Raised if more than one FDS input file found in specified directory
            Raised if no input file present and CHID could not be determined from results file names

        """
        self.workdir_path = results_dir
        self.upload_files = upload_files
        self.slice_parse_quantity = slice_parse_quantity
        self.slice_parse_ignore_zeros = slice_parse_ignore_zeros
        self._loading_historic_run = True

        # Find input file inside results dir
        _fds_files = list(pathlib.Path(results_dir).rglob("*.fds"))

        if not _fds_files:
            # Give a warning that no input file was found
            logger.warning(
                "No FDS input file found in your results directory - input metadata will not be stored."
            )

            # Try to deduce CHID by common prefix within directory
            all_files = [
                file.name
                for file in pathlib.Path(results_dir).iterdir()
                if file.is_file()
            ]
            self._chid = os.path.commonprefix(all_files)
            if not self._chid:
                raise ValueError(
                    "Could not determine CHID from results directory due to files with inconsistent names."
                )

        elif len(_fds_files) > 1:
            raise ValueError(
                "Found more than one input '.fds' file - please make sure only one such file is in the results directory."
            )
        else:
            self.fds_input_file_path = _fds_files[0]
            self.save_file(self.fds_input_file_path, "input")

            # Load input file, upload as metadata
            _nml = f90nml.read(self.fds_input_file_path).todict()
            self._chid = _nml["head"]["chid"]
            self.update_metadata({"input_file": _nml})

            if (
                self.slice_parse_quantity
                and self.fds_input_file_path.stem != self._chid
            ):
                logger.warning(
                    "Detected FDS input file with name different to CHID - creating a copy for slice parser..."
                )
                shutil.copy(
                    self.fds_input_file_path,
                    pathlib.Path(results_dir).joinpath(f"{self._chid}.fds"),
                )

        self._results_prefix = str(pathlib.Path(results_dir).joinpath(self._chid))

        # Read relevant files and call methods directly
        # Will make it so that if there are files missing, the code will still upload whichever files it can...

        # Extract metadata and metrics from log (.out) file
        if pathlib.Path(f"{self._results_prefix}.out").exists():
            _data, _meta = self._header_metadata(
                input_file=f"{self._results_prefix}.out"
            )
            self.update_metadata({**_data, **_meta})

            with open(f"{self._results_prefix}.out", "r") as log_file:
                _, _log_metrics = self._log_parser(file_content=log_file.read())
            for _metric in _log_metrics:
                self._metrics_callback(
                    data=_metric, meta={"file_name": f"{self._results_prefix}.out"}
                )
        else:
            # If file was not found, no other way to obtain timestamps from when the simulation will run
            # Will default to using the last time the input file was edited for any time (t >= 0 ), with a warning
            logger.warning(
                "Warning: No '.out' file was found! You will be missing important metrics from your simulation.",
                "Cannot determine timestamps accurately - defaulting to last time the input file was modified.",
            )
        if not self._timestamp_mapping.size:
            self._timestamp_mapping = numpy.array(
                [
                    [
                        -1,
                        self.fds_input_file_path.stat().st_mtime
                        if self.fds_input_file_path
                        else datetime.now().timestamp(),
                    ]
                ]
            )

        # Extract metrics from DEVC and HRR files
        for _suffix in ("hrr", "devc"):
            if pathlib.Path(f"{self._results_prefix}_{_suffix}.csv").exists():
                with open(f"{self._results_prefix}_{_suffix}.csv", "r") as _file:
                    # Skip the first line, as that contains units and not the header names we want
                    next(_file)
                    for _step, _metric in enumerate(csv.DictReader(_file)):
                        _metric["step"] = _step
                        self._metrics_callback(
                            data=_metric,
                            meta={"file_name": f"{self._results_prefix}_{_suffix}.csv"},
                        )

        # Extract events / metadata from CTRL log file
        if pathlib.Path(f"{self._results_prefix}_devc_ctrl_log.csv").exists():
            with open(f"{self._results_prefix}_devc_ctrl_log.csv", "r") as ctrl_file:
                for _metric in csv.DictReader(ctrl_file):
                    _metric = {
                        key: val.replace(" ", "") for key, val in _metric.items() if val
                    }
                    self._ctrl_log_callback(data=_metric)

        if self.slice_parse_quantity:
            if not _fds_files:
                logger.warning(
                    "Slice cannot be parsed without an input file available - slice parsing disabled."
                )
            elif not list(pathlib.Path(results_dir).rglob("*.xyz")):
                logger.warning(
                    "No XYZ files detected in results directory - slice parsing disabled."
                )
            else:
                self._parse_slice()

        self._post_simulation()
