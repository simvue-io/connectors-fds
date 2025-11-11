"""Smokeview extension to FDS connector."""

import os
import pathlib
import time
import pydantic
import typing
import subprocess
import shutil
import re


class SimvueSmokeviewConfig(pydantic.BaseModel):
    smoke_types: (
        list[
            typing.Literal[
                "TEMPERATURE", "SOOT DENSITY", "CARBON DIOXIDE DENSITY", "HRRPUV"
            ],
        ]
        | None
    ) = None
    configuration_file: pydantic.FilePath | None = None
    load_particles: bool = False

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid"
    )


class Smokeview(pydantic.BaseModel):
    show_logs: bool = False
    fds_input_file: pydantic.FilePath
    results_directory: pydantic.DirectoryPath
    output_directory: pydantic.DirectoryPath = pathlib.Path.cwd()
    session_file_read: bool = False
    simvue_sv_config: SimvueSmokeviewConfig = SimvueSmokeviewConfig()
    _s3_last_modified: float = 0

    def _read_n_frames(self, input_file: pathlib.Path) -> int | None:
        """Read available frame count from FDS file."""
        with input_file.open() as in_f:
            _content = in_f.read()

        if not (_frame_count := re.findall(r"NFRAMES=(\d+)", _content)):
            return None
        return int(_frame_count[0])

    def _render_script(self, frame_index: int) -> str:
        _out_dir = self.output_directory.joinpath("rendered_images")
        _out_dir.mkdir(exist_ok=True)
        _out_str = "RENDERDIR\n"
        _out_str += f"\t{_out_dir.absolute()}\n"

        if self.simvue_sv_config.configuration_file:
            _out_str += "LOADINIFILE\n"
            _out_str += f"\t{self.simvue_sv_config.configuration_file.absolute()}\n"

        if list(self.results_directory.glob("*.s3d")):
            for smoke_type in self.simvue_sv_config.smoke_types or []:
                _out_str += "LOAD3DSMOKE\n"
                _out_str += f"\t{smoke_type}\n"

        if self.simvue_sv_config.load_particles:
            _out_str += "LOAD3DPARTICLES\n"

        _out_str += "SETTIMEVAL\n"
        _out_str += f"\t{frame_index}\n"
        _out_str += "RENDERONCE\n"
        _out_str += f"\t{self.fds_input_file.stem}_{frame_index}\n"
        _out_str += "EXIT\n"

        return _out_str

    @classmethod
    @pydantic.validate_call
    def execute_session_file(
        cls, session_file: pydantic.FilePath, *, show_logs: bool = False
    ) -> None:
        """Create SmokeView instance from session file."""
        if not (_smokeview_exe := shutil.which("smokeview")):
            raise FileNotFoundError("Could not find SmokeView binary.")

        _args: dict[str, typing.Any] = {}

        if not show_logs:
            _args["stdout"] = subprocess.DEVNULL
            _args["stderr"] = subprocess.DEVNULL

        subprocess.check_call(
            [f"{_smokeview_exe}", "-runscript", session_file.stem],
            cwd=session_file.parent,
            **_args,
        )

    @property
    def smokeview_last_update(self) -> float | None:
        """Returns when Smokeview last updated files."""
        _smokeview_smoke_file = list(self.results_directory.glob("*.s3d"))
        if not _smokeview_smoke_file:
            return
        return os.path.getmtime(f"{_smokeview_smoke_file[0]}")

    @pydantic.validate_call
    def snapshot(self, frame_index: pydantic.NonNegativeInt) -> pathlib.Path | None:
        """Take a snapshot of the simulation."""
        # Limit wait for s3d file update to 10 seconds
        # else assume frame index unavailable
        _wait_time: int = 0

        while (
            not self.smokeview_last_update
            or self.smokeview_last_update <= self._s3_last_modified
        ) and _wait_time < 10:
            _wait_time += 1
            time.sleep(1)

        if not self.smokeview_last_update:
            return

        self._s3_last_modified = self.smokeview_last_update

        # If the request exceeds specified frame limit abort
        _n_frames: int | None = self._read_n_frames(self.fds_input_file)

        if _n_frames and frame_index >= _n_frames:
            return None

        # Simulation has not progressed
        try:
            _smv_file = next(self.results_directory.rglob("*.smv"))
        except StopIteration:
            return None

        _session_file = self.results_directory.joinpath(f"{_smv_file.stem}.ssf")
        with _session_file.open("w") as out_f:
            _ = out_f.write(self._render_script(frame_index))

        self.execute_session_file(_session_file, show_logs=self.show_logs)

        _files = list(
            self.output_directory.joinpath("rendered_images").glob(
                f"{self.fds_input_file.stem}_{frame_index}*.png"
            )
        )

        if not _files:
            raise RuntimeError("No snapshot taken.")

        return _files[0]
