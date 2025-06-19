"""FDS Connector.

This module provides functionality for using Simvue to track and monitor an FDS (Fire Dynamics Simulator) simulation.
"""

import glob
import pathlib
import platform
import re
import resource
import shutil
import threading
import time
import typing

import click
import f90nml
import multiparser.parsing.file as mp_file_parser
import multiparser.parsing.tail as mp_tail_parser
import numpy
import pydantic
import pyfdstools
import simvue
from loguru import logger
from simvue_connector.connector import WrappedRun
from simvue_connector.extras.create_command import format_command_env_vars

from simvue_fds.utils import HiddenPrints


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
    _patterns: typing.List[typing.Dict[str, typing.Pattern]] = [
        {"pattern": re.compile(r"\s+Time\sStep\s+(\d+).*"), "name": "step"},
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

        for line in file_content.split("\n"):
            for pattern in self._patterns:
                match = pattern["pattern"].search(line)
                if match:
                    if pattern["name"] == "mesh":
                        _current_mesh = match.group(1)
                        break

                    if pattern["name"] == "step":
                        if _out_record:
                            _out_data += [_out_record]
                        _out_record = {}
                        _current_mesh = None

                    _metric_name = pattern["name"]
                    if _current_mesh:
                        _metric_name = f"{_metric_name}.mesh.{_current_mesh}"

                    _out_record[_metric_name] = match.group(1)

                    if pattern["name"] == "time":
                        self.log_event(
                            f"Time Step: {_out_record['step']}, Simulation Time: {_out_record['time']} s"
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
        self.log_metrics(
            data,
            timestamp=meta["timestamp"].replace(" ", "T"),
            time=metric_time,
            step=metric_step,
        )
        # Since we don't have 'step' information from CSV files, just increment on each reading, starting from 0
        self._step_tracker[meta["file_name"]] = int(metric_step) + 1

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

        self.log_event(event_str)
        self.update_metadata({data["ID"]: state})

    def _slice_parser(self):
        """Read and process all 2D slice files, uploading min, max and mean as metrics."""
        processed_time = -1
        step = 0
        while True:
            time.sleep(60 * self.slice_parse_interval)

            # grid_abs is an array of all possible grid points, shape (X, Y, Z, 3)
            # data_abs is an array of all values, shape (X, Y, Z, times)
            # times_out is an array of in simulation times
            try:
                # Need to silence this function so it doesnt print tons of junk
                with HiddenPrints():
                    grid_abs, data_abs, times_out = pyfdstools.readSLCF2Ddata(
                        self._chid,
                        str(pathlib.Path(self.workdir_path).absolute())
                        if self.workdir_path
                        else str(pathlib.Path.cwd()),
                        self.slice_parse_quantity,
                    )
            except ValueError as e:
                logger.error(
                    "Failed to collect 2D slice data - check that your slice quantity is valid. Slice parsing is disabled for this run."
                )
                break

            # Remove times which we have already processed
            times_out = times_out[: data_abs.shape[-1]]
            to_process = numpy.where(times_out > processed_time)[0]

            if len(to_process) == 0:
                continue

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

            def add_metrics(
                metrics: dict,
                sub_slice: numpy.ndarray,
                label: str,
                name: float,
                ignore_zeros: bool,
            ):
                sub_slice = sub_slice[~numpy.isnan(sub_slice)]
                if ignore_zeros:
                    sub_slice = sub_slice[numpy.where(sub_slice != 0)]
                metrics[
                    f"{self.slice_parse_quantity.replace(' ', '_').lower()}.{label}.{str(name).replace('.', '_')}.min"
                ] = numpy.min(sub_slice)
                metrics[
                    f"{self.slice_parse_quantity.replace(' ', '_').lower()}.x.{str(name).replace('.', '_')}.max"
                ] = numpy.max(sub_slice)
                metrics[
                    f"{self.slice_parse_quantity.replace(' ', '_').lower()}.x.{str(name).replace('.', '_')}.avg"
                ] = numpy.mean(sub_slice)
                return metrics

            for time_idx, time_val in enumerate(times_out):
                metrics = {}
                for idx in range(len(x_indices)):
                    sub_slice = x_slices[idx, :, :, time_idx]
                    metrics = add_metrics(
                        metrics,
                        sub_slice,
                        label="x",
                        name=x_names[idx],
                        ignore_zeros=self.slice_parse_ignore_zeros,
                    )

                for idx in range(len(y_indices)):
                    sub_slice = y_slices[:, idx, :, time_idx]
                    metrics = add_metrics(
                        metrics,
                        sub_slice,
                        label="y",
                        name=y_names[idx],
                        ignore_zeros=self.slice_parse_ignore_zeros,
                    )

                for idx in range(len(z_indices)):
                    sub_slice = z_slices[:, :, idx, time_idx]
                    metrics = add_metrics(
                        metrics,
                        sub_slice,
                        label="z",
                        name=z_names[idx],
                        ignore_zeros=self.slice_parse_ignore_zeros,
                    )

                self.log_metrics(metrics, time=float(time_val), step=step)
                step += 1

            processed_time = times_out[-1]

            if self._trigger.is_set():
                break

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

        # Should add locking to this? And thread each call individuall to make sure it frees memory? #TODO
        if self.slice_parse_quantity:
            slice_parser = threading.Thread(target=self._slice_parser)
            slice_parser.daemon = True  # TODO: is this safe in this case? Have done it so that ctrl C doesnt wait for sleep to finish
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
        slice_parse_quantity: str = None,
        slice_parse_interval: int = 1,
        slice_parse_ignore_zeros: bool = True,
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
        slice_parse_quantity: str, optional
            ***** WARNING: EXPERIMENTAL FEATURE*****
            The quantity for which to find any 2D slices saved by the simulation, and upload the min/max/average as metrics
            Default is None, which will disable this feature
        slice_parse_interval : int, optional
            Interval (in minutes) at which to parse and upload 2D slice data, default is 1
        slice_parse_ignore_zeros : bool, optional
            Whether to ignore values of zero in slices (useful if there are obstructions in the mesh), default is True
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
        self._slice_time = -1
        self._step_tracker = {}

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
            if not nml["dump"]["write_xyz"]:
                raise ValueError(
                    "WRITE_XYZ must be enabled in your FDS file for slice parsing."
                )

        super().launch()
