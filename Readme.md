# ManoptExamples.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![CI](https://github.com/JuliaManifolds/ManoptExamples.jl/workflows/CI/badge.svg)](https://github.com/JuliaManifolds/ManoptExamples.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/JuliaManifolds/ManoptExamples.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaManifolds/ManoptExamples.jl)


This package provides examples of optimizations problems on Riemannian manifolds.

It uses the manifolds from [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/) and the structures provided by [Manopt.jl](https://manoptjl.org/).
to state the problems.
Furthermore the problems are illustrated in their usage in [Quarto](https://quarto.org) Markdown files, that
are pre-rendered into the documentation. This way, the package also provides examples of
the provided examples “in action”.

## Citation

If you use `ManoptExamples.jl` in your work, please cite `Manopt.jl`, i.e. the following

```biblatex
@article{Bergmann2022,
    Author    = {Ronny Bergmann},
    Doi       = {10.21105/joss.03866},
    Journal   = {Journal of Open Source Software},
    Number    = {70},
    Pages     = {3866},
    Publisher = {The Open Journal},
    Title     = {Manopt.jl: Optimization on Manifolds in {J}ulia},
    Volume    = {7},
    Year      = {2022},
}
```

To refer to a certain version or the source code in general we recommend to cite for example

```biblatex
@software{manoptjl-zenodo-mostrecent,
    Author = {Ronny Bergmann},
    Copyright = {MIT License},
    Doi = {10.5281/zenodo.4290905},
    Publisher = {Zenodo},
    Title = {Manopt.jl},
    Year = {2022},
}
```

for the most recent version or a corresponding version specific DOI, see [the list of all versions](https://zenodo.org/search?page=1&size=20&q=conceptrecid:%224290905%22&sort=-version&all_versions=True).
Note that both citations are in [BibLaTeX](https://ctan.org/pkg/biblatex) format.