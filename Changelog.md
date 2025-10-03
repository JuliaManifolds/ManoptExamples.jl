# Changelog

All notable changes to this Julia package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.16] – 03/10/2025

### Added

* Two numerical experiments from the Proximal Gradient preprints.

## [0.1.15] – 18/08/2025

### Added

* Numerical experiments from the Proximal Gradient preprints.

## [0.1.14] – 14/04/2025

### Added

* Projected Gradient example.

## [0.1.13] – 21/03/2025

### Changed

* Updated numerical experiments from the Riemannian Convex Bundle paper.

## [0.1.12] – 11/02/2025

### Changed

* Update ManifoldDiff.jl compat to 0.4.

## [0.1.11] – 10/02/2025

### Changed

* Bump dependencies on CI
* Adapt to ManifoldsBase 1.0

## [0.1.10] – 18/10/2024

### Changed

* Bump dependencies

## [0.1.9] – 28/06/2024

### Added

* Three numerical experiments from the Riemannian Convex Bundle paper.

## [0.1.8] – 12/06/2024

### Changed

* use `range` compatible with Julia 1.6 and hence lower the compatibility entry for Julia again.

## [0.1.7] – 07/06/2024

### Changed

* make `Manopt.jl` a weak dependency and load functions that require parts of it
  only load as an extension. This makes it easier to use the examples in the tests
  of Manopt itself.

## [0.1.6] – 22/03/2024

### Added

* Hyperparameter optimization example.

## [0.1.3] – 11/12/2023

### Added

* Total variation Minimization cost, proxes, and an example
* Bézier curve cost, gradients, and an example.

## [0.1.3] – 16/09/2023

### Added

* Rayleigh Quotient functions added
* an example illustrating Euclidean gradient/HEssian conversion
* Add Literature with DocumenterCitations

## [0.1.2] – 13/06/2023

### Added

* Update examples to use Quarto
* Add DC examples

## [0.1.1] – 01/03/2023

### Added

* Rosenbrock function and examples

## [0.1.0] – 18/02/2023

### Added

* Setup the project to collect example objectives for [Manopt.jl](https://manopt.org) which are well-documented and well-tested
* Setup Documentation to provide one example Quarto file for every example objective to illustrate how to use them.
