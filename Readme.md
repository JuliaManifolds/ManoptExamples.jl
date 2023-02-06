# ManoptExamples.jl

This package provides examples of optimizations problems on Riemannian manifolds.

It uses the manifolds from [Manifolds.jl]https://juliamanifolds.github.io/Manifolds.jl/) and the structures provided by [Manopt.jl](https://manoptjl.org/).
to state the problems.
Furthermore the problems are illustrated in their usage in [Quarto]() Markdown files, that
are pre-rendered into the documentation. This way, the package also provides examples of
the provided examples “in action”.

## Roadmap

This package is still in a little preliminary state. We still need

* [ ] to setup the documentation and its CI
* [ ] to setup the test suite and its CI
* [ ] to setup the formatter CI test run.
* [ ] to setup the quarto environment
* [ ] to write a first example problem, probably just the mean to illustrate how this package is intended
* [ ] to write a first Quarto example (on how to write examples)
* [ ] to setup zenodo for this repo as well to keep the version persistently archived.

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