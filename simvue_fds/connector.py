"""FDS Connector.

This module provides functionality for using Simvue to track and monitor an FDS (Fire Dynamics Simulator) simulation.
"""

import csv
import glob
import os
import pathlib
import platform
import re
import shlex
import shutil
import threading
import time
import typing
from datetime import datetime, timezone

import pandas

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import logging

import click
import f90nml
import fdsreader
import multiparser.parsing.file as mp_file_parser
import multiparser.parsing.tail as mp_tail_parser
import numpy
import pydantic
import simvue
from fdsreader.slcf.slice import Slice
from simvue_connector.connector import WrappedRun
from simvue_connector.extras.create_command import format_command_env_vars

logger = logging.getLogger(__name__)
MAXIMUM_SLICE_SIZE: int = 50000


class FDSRun(WrappedRun):
    """Class for setting up Simvue tracking and monitoring of an FDS simulation.

    Use this class as a context manager, in the same way you use default Simvue runs, and call run.launch(). Eg:

    with FDSRun() as run:
        run.init(
            name="fds_simulation",
        )
        run.launch(...)
    """

    _patterns: list[dict[str, typing.Pattern]] = [
        {"pattern": re.compile(r"\s+Time\sStep\s+(\d+)\s+([\w\s:,]+)"), "name": "step"},
        {
            "pattern": re.compile(r"\s+Step\sSize:.*Total\sTime:\s+([\d\.\-]+)\ss.*"),
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
            self._sv_obj.started = datetime.fromtimestamp(
                self._timestamp_mapping[0, 1], tz=timezone.utc
            )
            self._sv_obj.endtime = datetime.fromtimestamp(
                self._timestamp_mapping[-1, 1], tz=timezone.utc
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
        return datetime.fromtimestamp(_timestamp, tz=timezone.utc)

    def _find_fds_executable(self) -> str:
        """Locates the FDS executable on the user's system.

        Returns
        -------
        str
            The path to the executable

        Raises
        ------
        EnvironmentError
            FDS executable could not be found

        """
        fds_bin = None
        # Set stack limit - analogous to 'ulimit -s' recommended in FDS documentation
        if platform.system() != "Windows":
            import resource

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
            # Find path to FDS executable
            fds_bin = shutil.which("fds")

        else:
            for search_loc in (
                pathlib.Path(os.environ["PROGRAMFILES"]).joinpath("firemodels"),
                pathlib.Path(os.environ["LOCALAPPDATA"]).joinpath("firemodels"),
                pathlib.Path.home().joinpath("firemodels"),
            ):
                if not search_loc.exists():
                    continue
                if search := pathlib.Path(search_loc).rglob("**/fds_local.bat"):
                    fds_bin = f"{next(search)}"
                    break

        if not fds_bin:
            raise EnvironmentError("FDS executable could not be found!")

        return fds_bin

    def _map_line_var_coords(self) -> None:
        """Map DEVC line variables to their coordinates."""
        self._line_var_coords = {}
        _labels = ("x", "y", "z")
        # Loop through input dict and find all DEVC devices
        for key, devc in self._input_dict.items():
            if "devc" not in key:
                continue
            # If devc device does not have an ID, cannot add to mapping, so ignore
            if not (_devc_id := devc.get("id")):
                continue
            # Figure out which axes it varies along
            _devc_coords: list[float] = devc.get("xb") or devc.get("xbp")
            if not _devc_coords:
                continue

            _devc_coords_change = [
                _devc_coords[1] - _devc_coords[0],
                _devc_coords[3] - _devc_coords[2],
                _devc_coords[5] - _devc_coords[4],
            ]

            _devc_coords_idx = [
                idx for idx, val in enumerate(_devc_coords_change) if abs(val) > 0
            ]

            if len(_devc_coords_idx) == 0:
                # Indicates a point DEVC device using XB or XBP definition - dont add to mapping, just continue
                continue

            elif len(_devc_coords_idx) > 1:
                # Not varying in 1D, currently not supported # TODO
                continue
            _devc_coord_label = _labels[_devc_coords_idx[0]]
            _devc_coord_id = devc.get(f"{_devc_coord_label}_id")
            # Check if specific points chosen by user
            _axes_ticks = devc.get(f"points_array_{_devc_coord_label}")
            if not _axes_ticks:
                # Create linear spacing
                _low = _devc_coords[_devc_coords_idx[0] * 2]
                _high = _devc_coords[_devc_coords_idx[0] * 2 + 1]
                _num = devc.get("points")
                _axes_ticks = numpy.linspace(_low, _high, _num)

                # If D_ID provided, user wants ticks in terms of distance from first point
                if d_id := devc.get("d_id"):
                    _axes_ticks = _axes_ticks - _axes_ticks[0]
                    _devc_coord_id = d_id

                # If R_ID provided, user wants ticks in terms of distance from the origin
                elif r_id := devc.get("r_id"):
                    _fixed_dims = [0, 1, 2]
                    _fixed_dims.remove(_devc_coords_idx[0])
                    _fixed_dim_positions = [
                        _devc_coords[idx * 2] for idx in _fixed_dims
                    ]
                    _axes_ticks = numpy.sqrt(
                        _axes_ticks**2 + sum(val**2 for val in _fixed_dim_positions)
                    )
                    _devc_coord_id = r_id

            self._line_var_coords[_devc_id] = {
                "label": _devc_coord_id or _devc_coord_label,
                "ticks": list(_axes_ticks),
            }

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
                        _out_record = {"timestamp": _timestamp}
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
                            timestamp=_out_record["timestamp"]
                            if self._loading_historic_run
                            else datetime.now(timezone.utc),
                        )
                        # If loading from historic runs, keep track of time to timestamp mapping
                        if self._loading_historic_run:
                            self._timestamp_mapping = numpy.vstack(
                                (
                                    self._timestamp_mapping,
                                    [
                                        float(_out_record["time"]),
                                        _out_record["timestamp"].timestamp(),
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

    def _input_file_callback(self, data: dict, meta: dict):
        self._input_dict = {k: v for k, v in data.items() if v}
        self.update_metadata({"input_file": self._input_dict})

    def _metrics_callback(self, data: dict, meta: dict) -> None:
        """Log metrics extracted from a log file to Simvue.

        Parameters
        ----------
        data : dict
            Dictionary of data to log to Simvue as metrics
        meta: dict
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
            metric_timestamp = data.pop("timestamp", datetime.now(timezone.utc))

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

    def _ctrl_log_callback(self, data: dict, *_) -> None:
        """Log metrics extracted from the CTRL log file to Simvue.

        Parameters
        ----------
        data : dict
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

        self.update_metadata({str(data["ID"]): state})

    def _line_parser(
        self, input_file: str, **__
    ) -> tuple[dict, dict[str, list[float]]]:
        df = pandas.read_csv(input_file, skiprows=1)
        return {}, df.to_dict(orient="list")

    def _line_callback(self, data, meta) -> None:
        # Generate devc to coord mapping if it doesnt exist:
        if self._line_var_coords is None:
            self._map_line_var_coords()

        # Create metric data
        _metric_data = {}
        # For each key, check if it is a DEVC device we can track
        for key, values in data.items():
            if not (_axes := self._line_var_coords.get(key)):
                continue

            metric = numpy.array(values)
            metric = metric[~numpy.isnan(metric)]

            # Add a catch here to check if removal of NaNs has caused the data to be a different length to the axes
            if metric.shape[0] != len(_axes["ticks"]):
                logger.warning(
                    f"Unexpected NaN value found in line metric '{key}': This metric will not be recorded."
                )
                self._line_var_coords.pop(key)
                continue

            # Assign to grid if required
            if key not in self._grids.keys():
                self.assign_metric_to_grid(
                    metric_name=key,
                    axes_ticks=[_axes["ticks"]],
                    axes_labels=[_axes["label"]],
                )

            if numpy.any(metric):
                _metric_data[key] = metric
        if _metric_data:
            # Time is fixed to 1, since we have no way of knowing at which time line devices were recorded
            _metric_data["time"] = 1
            self._metrics_callback(_metric_data, meta)

    def _parse_slice(self) -> bool:
        """Parse slices present in the FDS results files and extract data as metrics.

        Returns
        -------
        bool
            Whether the slice was successfuly extracted

        """
        sim_dir = self.workdir_path if self.workdir_path else pathlib.Path.cwd()
        try:
            sim = fdsreader.Simulation(str(sim_dir.absolute()))
        except OSError as e:
            logger.warning(
                f"""
                Unable to load slice data found in output directory '{sim_dir}'. Slice parsing is disabled for this run.
                This is because: {e}
                Please correct the issue described above, or raise a bug report via the UI if you think this is incorrect.
                """
            )
            return False
        slices: list[Slice] = (
            [sim.slices.get_by_id(_id) for _id in self.slice_parse_ids]
            if self.slice_parse_ids
            else sim.slices
        )
        # Get rid of any Nones - caused by slice IDs not found in results
        slices = [slice for slice in slices if slice is not None]

        if self.slice_parse_quantities:
            slices = [
                slice
                for slice in slices
                if slice.quantity.quantity in self.slice_parse_quantities
            ]

        if self.slice_parse_fixed_dimensions:
            slices = [
                slice
                for slice in slices
                if any(
                    dim not in slice.extent_dirs
                    for dim in self.slice_parse_fixed_dimensions
                )
            ]

        # Calculate the metrics which need to be sent, store in format:
        # {time: {metric_name: [], timestamp: ""}}
        slice_metrics: dict[float, dict] = {}
        times = None

        for slice in slices:
            if not slice:
                continue
            # Check it is a 2D slice, if not then ignore
            if slice.type == "3D":
                continue

            # Get the values, coordinates, times
            # Due to edge cases which may break fdsreader, we cover this in a try... except
            try:
                values, coords = slice.to_global(
                    masked=True, fill=numpy.nan, return_coordinates=True
                )
                times = slice.times
            except Exception as e:
                if not self._grids_defined:
                    logger.warning(
                        "Unable to parse a slice due to unexpected values within the slice - enable debug logging for full traceback."
                    )
                    logger.debug(e)
                continue

            # Get rid of values already uploaded, return if nothing left to upload
            values = values[self._slice_processed_idx :, ...]
            if values.shape[0] == 0:
                continue

            # Check if slice has an ID, generate a metric name if not
            metric_name = slice.id
            if not metric_name:
                # Will name it {quantity}.{axis}.{value}
                quantity = slice.quantity.quantity.replace(" ", "_").lower()
                axis = next(ax for ax in ("x", "y", "z") if ax not in slice.extent_dirs)
                value = str(round(coords[axis][0], 3)).replace(".", "_")
                metric_name = f"{quantity}.{axis}.{value}"

            # If grid is too large, just go to next one as not tracking
            if metric_name in self._grids_too_large:
                continue

            # Define grid if first pass
            if metric_name not in self._grids_defined:
                # Check size doesn't breach server limit
                if (
                    coords[slice.extent_dirs[0]].shape[0]
                    * coords[slice.extent_dirs[1]].shape[0]
                    > MAXIMUM_SLICE_SIZE
                ):
                    logger.warning(
                        f"Slice '{metric_name}' exceeds the maximum size for upload to the server - ignoring this metric."
                    )
                    self._grids_too_large.append(metric_name)
                    continue

                self.assign_metric_to_grid(
                    metric_name=metric_name,
                    axes_ticks=[
                        coords[slice.extent_dirs[0]].tolist(),
                        coords[slice.extent_dirs[1]].tolist(),
                    ],
                    axes_labels=slice.extent_dirs,
                )
                self._grids_defined.append(metric_name)

            times_to_process = times[self._slice_processed_idx :]
            for time_idx, time_val in enumerate(times_to_process):
                values_at_time = values[time_idx, ...]
                values_no_obst = values_at_time[~numpy.isnan(values_at_time)]

                slice_metrics.setdefault(time_val, {})
                if metric_name in self._grids_defined:
                    slice_metrics[time_val].update(
                        {
                            metric_name: numpy.nan_to_num(values_at_time).T,
                            f"{metric_name}.min": numpy.min(values_no_obst),
                            f"{metric_name}.max": numpy.max(values_no_obst),
                            f"{metric_name}.avg": numpy.mean(values_no_obst),
                        }
                    )

                # Need to estimate timestamp which this measurement would correspond to
                # Will use estimate = timestamp of last parse + (now - last parse) * (idx/len(times_out))
                if not slice_metrics[time_val].get("timestamp"):
                    slice_metrics[time_val]["timestamp"] = self._last_parse_time + (
                        datetime.now(timezone.utc).timestamp() - self._last_parse_time
                    ) * ((time_idx + 1) / len(times_to_process))

        for time_val, metrics in slice_metrics.items():
            timestamp = metrics.pop("timestamp", None)
            self.log_metrics(
                metrics,
                time=time_val,
                step=self._slice_step,
                timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc)
                if timestamp
                else None,
            )
            self._slice_step += 1

        self._last_parse_time = datetime.now(timezone.utc).timestamp()
        self._slice_processed_idx = (
            times.shape[0] if times is not None else self._slice_processed_idx
        )
        return True

    def _slice_parser(self) -> None:
        """Read and process all 2D slice files in a loop, uploading min, max and mean as metrics."""
        while True:
            self._trigger.wait(timeout=self.slice_parse_interval)
            slice_parsed = self._parse_slice()

            if self._trigger.is_set() or not slice_parsed:
                break

    def __init__(
        self,
        mode: typing.Literal["online", "offline", "disabled"] = "online",
        abort_callback: typing.Optional[typing.Callable[[Self], None]] = None,
        server_token: typing.Optional[str] = None,
        server_url: typing.Optional[str] = None,
        debug: bool = False,
    ):
        """Initialize the FDSRun instance.

        If `abort_callback` is provided the first argument must be this Run instance.

        Parameters
        ----------
        mode : typing.Literal['online', 'offline', 'disabled'], optional
            mode of running, by default 'online':
                online - objects sent directly to Simvue server
                offline - everything is written to disk for later dispatch
                disabled - disable monitoring completelyby default "online"
        abort_callback : typing.Optional[typing.Callable[[Self], None]], optional
            callback executed when the run is aborted, by default None
        server_token : typing.Optional[str], optional
            overwrite value for server token, by default None
        server_url : typing.Optional[str], optional
            overwrite value for server URL, by default None
        debug : bool, optional
            run in debug mode, by default False

        """
        self.fds_input_file_path: pathlib.Path = None
        self.workdir_path: pathlib.Path | None = None
        self.upload_files: list[str] = None
        self.slice_parse_enabled: bool = False
        self.slice_parse_ids: list[str] | None = None
        self.slice_parse_interval: int = 60
        self.ulimit: str | int = None
        self.fds_env_vars: dict[str, typing.Any] = None

        # Users can set this before launching a simulation, if they want (not in launch to not bloat arguments required)
        self.upload_input_file: bool = True

        self._slice_processed_idx: int = 0
        self._slice_step: int = 0
        self._step_tracker: dict = {}
        self._last_parse_time: float = datetime.now(timezone.utc).timestamp()
        self._activation_times: bool = False
        self._activation_times_data: dict[str, float] = {}
        self._chid: str = ""
        self._results_prefix: str = ""
        self._loading_historic_run: bool = False
        self._timestamp_mapping: numpy.ndarray = numpy.empty((0, 2))
        self._input_dict: dict = {}
        self._line_var_coords: dict | None = None

        # Need this so that we dont get spammed with non-critical timestamp warning from fdsreader
        fdsreader.settings.IGNORE_ERRORS = True
        # Disable caching so that it doesnt create pickle files inside the results directory
        fdsreader.settings.ENABLE_CACHING = False

        super().__init__(
            mode=mode,
            abort_callback=abort_callback,
            server_token=server_token,
            server_url=server_url,
            debug=debug,
        )

    def _pre_simulation(self):
        """Start the FDS process."""
        super()._pre_simulation()
        self.log_event("Starting FDS simulation")

        # Save the FDS input file for this run to the Simvue server
        if self.upload_input_file and self.fds_input_file_path.exists():
            self.save_file(self.fds_input_file_path, "input")

        def check_for_errors(status_code, std_out, std_err):
            """Need to check for 'ERROR' in logs, since FDS returns rc=0 even if it throws an error."""
            self._trigger.set()
            if "ERROR" in std_err or status_code != 0:
                click.secho(
                    f"[simvue] Run failed - FDS encountered an error: {std_err}",
                    fg="red" if self._term_color else None,
                    bold=self._term_color,
                )
                self._failed = True
                self.log_event("FDS encountered an error:")
                self.log_event(std_err)
                if alert_id := self._executor._alert_ids.get("fds_simulation"):
                    self.log_alert(
                        identifier=alert_id,
                        state="critical",
                    )
            self.kill_all_processes()

        if run_command := os.getenv("SIMVUE_FDS_RUN_COMMAND"):
            logger.warning(
                "Custom FDS run command provided - environment variables passed into launch will be ignored."
            )
            command: list = shlex.split(
                run_command, posix=platform.system() == "Windows"
            )
            command.append(str(self.fds_input_file_path.absolute()))

        else:
            fds_bin = self._find_fds_executable()
            command = []
            if platform.system() == "Windows":
                if self.run_in_parallel:
                    command += [
                        f"{fds_bin}",
                        "-p",
                        str(self.num_processors),
                        str(self.fds_input_file_path),
                    ]
                else:
                    command += [f"{fds_bin}", str(self.fds_input_file_path.absolute())]
            else:
                if self.run_in_parallel:
                    command += ["mpiexec", "-n", str(self.num_processors)]
                    command += format_command_env_vars(self.mpiexec_env_vars)
                command += [f"{fds_bin}", str(self.fds_input_file_path.absolute())]

            command += format_command_env_vars(self.fds_env_vars)

        self.add_process(
            "fds_simulation",
            *command,
            cwd=self.workdir_path,
            completion_callback=check_for_errors,
        )

        if self.slice_parse_enabled:
            self.slice_parser = threading.Thread(
                target=self._slice_parser, daemon=True, name="slice_parser"
            )
            self.slice_parser.start()

        if self._concatenated_input_files:
            self.fds_input_file_path = pathlib.Path(f"{self._results_prefix}.fds")

    def _during_simulation(self):
        """Describe which files should be monitored during the simulation by Multiparser."""
        # Upload data from input file as metadata
        self.file_monitor.track(
            path_glob_exprs=str(self.fds_input_file_path),
            callback=self._input_file_callback,
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
        # Track line.csv file, can be entirely overwritten on each time step
        self.file_monitor.track(
            path_glob_exprs=f"{self._results_prefix}_line.csv",
            parser_func=mp_file_parser.file_parser(self._line_parser),
            callback=self._line_callback,
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

        # Upload updated FDS file if '&CATF' namespace in FDS file
        if self._concatenated_input_files:
            self.save_file(str(self.fds_input_file_path), "input")

        if self.upload_files is None:
            for file in glob.glob(f"{self._results_prefix}*"):
                if pathlib.Path(file).absolute() == self.fds_input_file_path.absolute():
                    continue
                self.save_file(file, "output")
        else:
            if self.workdir_path:
                self.upload_files = [
                    str(self.workdir_path.joinpath(path)) for path in self.upload_files
                ]

            for path in self.upload_files:
                for file in glob.glob(path):
                    if (
                        pathlib.Path(file).absolute()
                        == self.fds_input_file_path.absolute()
                    ):
                        continue
                    self.save_file(file, "output")

        # Then wait for slice parser to finish
        if self.slice_parser:
            self.slice_parser.join()
        super()._post_simulation()

    @simvue.utilities.prettify_pydantic
    @pydantic.validate_call
    def launch(
        self,
        fds_input_file_path: pydantic.FilePath,
        workdir_path: str | pathlib.Path = None,
        clean_workdir: bool = False,
        upload_files: list[str] | None = None,
        slice_parse_enabled: bool = False,
        slice_parse_quantities: list[str] | None = None,
        slice_parse_fixed_dimensions: list[typing.Literal["x", "y", "z"]] | None = None,
        slice_parse_ids: list[str] | None = None,
        slice_parse_interval: int = 60,
        ulimit: typing.Literal["unlimited"] | int = "unlimited",
        fds_env_vars: typing.Optional[dict[str, typing.Any]] = None,
        run_in_parallel: bool = False,
        num_processors: int = 1,
        mpiexec_env_vars: typing.Optional[dict[str, typing.Any]] = None,
    ):
        """Command to launch the FDS simulation and track it with Simvue.

        Parameters
        ----------
        fds_input_file_path : pydantic.FilePath
            Path to the FDS input file to use in the simulation
        workdir_path : str | pathlib.Path, optional
            Path to a directory which you would like FDS to run in, by default None
            This is where FDS will generate the results from the simulation
            If a directory does not already exist at this path, it will be created
            Uses the current working directory by default.
        clean_workdir : bool, optional
            Whether to remove all FDS related files from the working directory, by default False
            Useful when doing optimisation problems to remove results from previous runs.
        upload_files : list[str] | None, optional
            List of results file names to upload to the Simvue server for storage, by default None
            These should be supplied as relative to the working directory specified above (if specified, otherwise relative to cwd)
            If not specified, will upload all files by default. If you want no results files to be uploaded, provide an empty list.
        slice_parse_enabled: bool, optional
            Whether to enable slice parsing for this run, by default False
        slice_parse_quantities: list[str] | None, optional
            If slice parsing is enabled, upload all slices which are measuring one of these quantities. Default is None, which will upload all slices.
        slice_parse_fixed_dimensions: list[typing.Literal["x", "y", "z"]] | None, optional
            If slice parsing is enabled, the fixed dimension(s) which to upload slices for. Default is None, which will upload all slices.
            Note if this is specified along with other `slice_parse_` parameters, only slices which match all conditions will be uploaded.
        slice_parse_ids: list[str] | None, optional
            If slice parsing is enabled, the IDs of the slices to upload as Metrics. Default is None, which will upload all slices.
            Note if this is specified along with other `slice_parse_` parameters, only slices which match all conditions will be uploaded.
        slice_parse_interval : int, optional
            Interval (in seconds) at which to parse and upload 2D slice data, default is 60
        ulimit : typing.Literal["unlimited"] | int, optional
            Value to set your stack size to (for Linux and MacOS), by default "unlimited"
        fds_env_vars : typing.Optional[dict[str, typing.Any]], optional
            Environment variables to provide to FDS when executed, by default None
        run_in_parallel: bool, optional
            Whether to run the FDS simulation in parallel, by default False
        num_processors : int, optional
            The number of processors to run a parallel FDS job across, by default 1
        mpiexec_env_vars : typing.Optional[dict[str, typing.Any]]
            Any environment variables to pass to mpiexec on startup if running in parallel, by default None

        """
        self.fds_input_file_path = pathlib.Path(fds_input_file_path)
        self.workdir_path = pathlib.Path(workdir_path) if workdir_path else None
        self.upload_files = upload_files
        self.slice_parse_enabled = slice_parse_enabled
        self.slice_parse_quantities = slice_parse_quantities
        self.slice_parse_fixed_dimensions = slice_parse_fixed_dimensions
        self.slice_parse_ids = slice_parse_ids
        self.slice_parse_interval = slice_parse_interval
        self.slice_parser = None
        self.ulimit = ulimit
        self.fds_env_vars = fds_env_vars or {}
        self.run_in_parallel = run_in_parallel
        self.num_processors = num_processors
        self.mpiexec_env_vars = mpiexec_env_vars or {}

        self._activation_times = False
        self._activation_times_data = {}
        self._grids_defined = []
        self._grids_too_large = []

        logger.addHandler(simvue.Handler(self))

        nml = f90nml.read(self.fds_input_file_path).todict()
        self._chid = nml["head"]["chid"]

        # Need to find if this FDS input file will concatenate together other files
        self._concatenated_input_files = []
        for key in nml.keys():
            if "catf" in key:
                self._concatenated_input_files += nml[key]["other_files"]
        if self._concatenated_input_files:
            self._chid += "_cat"

        if self.workdir_path:
            self.workdir_path.mkdir(exist_ok=True)

            if clean_workdir:
                for file in self.workdir_path.glob(f"{self._chid}*"):
                    if (
                        pathlib.Path(file).absolute()
                        == self.fds_input_file_path.absolute()
                        or file.name in self._concatenated_input_files
                    ):
                        continue
                    pathlib.Path(file).unlink()

            # Copy files to concatenate together into working directory
            for concat_file in self._concatenated_input_files:
                concat_path = self.fds_input_file_path.parent.joinpath(concat_file)
                if (
                    concat_path.exists()
                    and concat_path.absolute()
                    != self.workdir_path.joinpath(concat_file).absolute()
                ):
                    shutil.copy(
                        concat_path,
                        self.workdir_path.joinpath(concat_file),
                    )

        self._results_prefix = (
            str(self.workdir_path.joinpath(self._chid))
            if self.workdir_path
            else self._chid
        )

        super().launch()

    def load(
        self,
        results_dir: pydantic.DirectoryPath,
        upload_files: list[str] | None = None,
        slice_parse_enabled: bool = False,
        slice_parse_quantities: list[str] | None = None,
        slice_parse_fixed_dimensions: list[typing.Literal["x", "y", "z"]] | None = None,
        slice_parse_ids: list[str] | None = None,
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
        slice_parse_enabled: bool, optional
            Whether to enable slice parsing for this run, by default False
        slice_parse_quantities: list[str] | None, optional
            If slice parsing is enabled, upload all slices which are measuring one of these quantities. Default is None, which will upload all slices.
        slice_parse_fixed_dimensions: list[typing.Literal["x", "y", "z"]] | None, optional
            If slice parsing is enabled, the fixed dimension(s) which to upload slices for. Default is None, which will upload all slices.
            Note if this is specified along with other `slice_parse_` parameters, only slices which match all conditions will be uploaded.
        slice_parse_ids: list[str] | None, optional
            If slice parsing is enabled, the IDs of the slices to upload as Metrics. Default is None, which will upload all slices.
            Note if this is specified along with slice_parse_quantities, only slices which match both conditions will be uploaded.

        Raises
        ------
        ValueError
            Raised if more than one FDS input file found in specified directory
            Raised if no input file present and CHID could not be determined from results file names

        """
        self.workdir_path = pathlib.Path(results_dir)
        self.upload_files = upload_files
        self.slice_parse_enabled = slice_parse_enabled
        self.slice_parse_quantities = slice_parse_quantities
        self.slice_parse_fixed_dimensions = slice_parse_fixed_dimensions
        self.slice_parse_ids = slice_parse_ids

        self.slice_parser = None
        self._loading_historic_run = True
        self._grids_defined = []
        self._grids_too_large = []
        self._concatenated_input_files = False

        logger.addHandler(simvue.Handler(self))

        # Find input file inside results dir
        _fds_files = list(pathlib.Path(results_dir).rglob("*.fds"))

        if not _fds_files:
            # Give a warning that no input file was found
            logger.warning(
                "No FDS input file found in your results directory - input metadata will not be uploaded."
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
            self._input_dict = f90nml.read(self.fds_input_file_path).todict()
            self._chid = self._input_dict["head"]["chid"]
            self.update_metadata({"input_file": self._input_dict})

            if self.slice_parse_enabled and self.fds_input_file_path.stem != self._chid:
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
                """Warning: No '.out' file was found! You will be missing important metrics from your simulation.
                Cannot determine timestamps accurately - defaulting to last time the input file was modified."""
            )
        if not self._timestamp_mapping.size:
            self._timestamp_mapping = numpy.array(
                [
                    [
                        -1,
                        self.fds_input_file_path.stat().st_mtime
                        if self.fds_input_file_path
                        else datetime.now(timezone.utc).timestamp(),
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

        # Extract line DEVC devices
        if pathlib.Path(f"{self._results_prefix}_line.csv").exists():
            if not _fds_files:
                logger.warning(
                    "Line DEVC devices cannot be parsed without an input file available."
                )
            else:
                self._map_line_var_coords()
                _, data = self._line_parser(f"{self._results_prefix}_line.csv")
                self._line_callback(
                    data, {"file_name": f"{self._results_prefix}_line.csv"}
                )

        if self.slice_parse_enabled:
            self._parse_slice()

        self._post_simulation()
