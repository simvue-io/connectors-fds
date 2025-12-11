# Change log

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
