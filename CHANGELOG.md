# Change log

## [v2.0.5](https://github.com/simvue-io/connectors-fds/releases/tag/v2.0.5) - 2026-02-06
* Fixes bug where slice parser crashes if uploading a slice located between boundaries

## [v2.0.4](https://github.com/simvue-io/connectors-fds/releases/tag/v2.0.3) - 2026-01-30
* Adds option to filter slices uploaded by their fixed dimension
* Adds support for FDS simulations with negative timesteps
* Fixes bug where slices may fail to upload a grid on the first iteration
* Makes slice parser error messages more descriptive

## [v2.0.3](https://github.com/simvue-io/connectors-fds/releases/tag/v2.0.4) - 2026-01-30
* Makes the slice parser retry if it fails to find simulation data initially

## [v2.0.2](https://github.com/simvue-io/connectors-fds/releases/tag/v2.0.2) - 2026-01-16
* Improves handling of error thrown by fdsreader.Simulation if no results are found
* Improves responsiveness of FDSRun() is the FDS simulation fails
* Fixes bugs related to passing paths to the input file or workdir as strings

## [v2.0.1](https://github.com/simvue-io/connectors-fds/releases/tag/v2.0.1) - 2026-01-16
* Fixes bug where slice parsing wouldn't work if a working dir not specified

## [v2.0.0](https://github.com/simvue-io/connectors-fds/releases/tag/v2.0.0) - 2026-01-15
* Makes slice parsing more efficient by using FDSReader
* Slices to parse can now be specified by ID as well as by quantities
* Breaking change to input format to launch and load commands
* Changes license to GPL3

## [v1.3.3](https://github.com/simvue-io/connectors-fds/releases/tag/v1.3.3) - 2026-01-15
* Fixes bug where IndexError is thrown if the FDS Simulation alert is not available
* Fixes bug caused by an edge case in FDS which gives NaNs in DEVC Line devices
* Adds environment variable to be able to specify the FDS run command manually

## [v1.3.2](https://github.com/simvue-io/connectors-fds/releases/tag/v1.3.2) - 2025-12-11
* Add zip of example results to releas artifacts

## [v1.3.1](https://github.com/simvue-io/connectors-fds/releases/tag/v1.3.1) - 2025-12-11
* Fixes bug with segfaulting FDS simulations hanging

## [v1.3.0](https://github.com/simvue-io/connectors-fds/releases/tag/v1.3.0) - 2025-12-11
* Adds support for DEVC line devices to be uploaded as 2D metrics

## [v1.2.1](https://github.com/simvue-io/connectors-fds/releases/tag/v1.2.1) - 2025-10-16
* Fixes bug with alerting due to timestamps not being sent in UTC
* Adds support for &CATF lines in FDS input files
* Fixes bug caused by zero-sized slice arrays in summary metric calculations

## [v1.2.0](https://github.com/simvue-io/connectors-fds/releases/tag/v1.2.0) - 2025-09-22
* Added functionality for uploading full 2D slices as 3D metrics
* Fixes a number of bugs for Windows users related to identifying correct FDS binary

## [v1.1.1](https://github.com/simvue-io/connectors-fds/releases/tag/v1.1.1) - 2025-06-27

* Added estimate of timestamps for each slice metric value to fix bug in Alert UI
* Added attribute which can be toggled to disable uploading of input file

## [v1.1.0](https://github.com/simvue-io/connectors-fds/releases/tag/v1.1.0) - 2025-06-26

* Added functionality to load historic FDS runs into Simvue
* Added functionality to parse 2D slices from FDS results and upload min, max, average as metrics


## [v1.0.0](https://github.com/simvue-io/connectors-fds/releases/tag/v1.0.0) - 2025-03-07

* Initial release of FDS Connector.
