# Contributing to `Manopt.jl`

First, thanks for taking the time to contribute.
Any contribution is appreciated and welcome.

The following is a set of guidelines to [`ManoptExamples.jl`](https://juliamanifolds.github.io/ManoptExamples.jl/).

#### Table of Contents

- [Contributing to `Manopt.jl`](#Contributing-to-manoptjl)
  - [Table of Contents](#Table-of-Contents)
  - [I just have a question](#I-just-have-a-question)
  - [How can I file an issue?](#How-can-I-file-an-issue)
  - [How can I contribute?](#How-can-I-contribute)
    - [Add a problem](#Add-a-problem)
  - [Code style](#Code-style)

## I just have a question

The developer can most easily be reached in the Julia Slack channel [#manifolds](https://julialang.slack.com/archives/CP4QF0K5Z).
You can apply for the Julia Slack workspace [here](https://julialang.org/slack/) if you haven't joined yet.
You can also ask your question on [our GitHub discussion](https://github.com/JuliaManifolds/ManoptExamples.jl/discussions).

## How can I file an issue?

If you found a bug or want to propose a feature, we track our issues within the [GitHub repository](https://github.com/JuliaManifolds/ManoptExamples.jl/issues).

## How can I contribute?

### Add a problem

If you have a sprecific problem you would like to provide here, feel free to start a new
file in the `src/problems/` folder in your own fork and propose it later as a [Pull Request](https://github.com/JuliaManifolds/ManoptExamples.jl/pulls).

If you problem works without reusing any other problems functions, then they can all just be placed in this one file.
If you notice, that you are reusing for example another problems gradient as part of your problem,
please refactor the code, such that the gradient, or other function is in the corresponding file in
`src/functions/` and follows the naming scheme:

* cost functions are always of the form `cost_` and a fitting name
* gradient functions are always of the the `gradient_` and a fitting name, followed by an `!`
for in-place gradients and by `!!` if it is a struct that can provide both.

It would be great if you could also add a small test for the functions and the problem you
defined in the `test/` section.

### Add an example

If you have used one of the problems from here in an example or you are providing a problem
together with an example, please add a correspoinding [Quarto]() Markdown file to the `examples/`
folder. The Markdown file should provide a short introduction to the problem and provide links
to further details, maybe a paper or a preprint. Use the `bib/literature.yaml` file to add
references (in [CSL_YAML](), which can for example be exported from [Zotero]()).

Add any packages you need to the `examples/` environment (see the containting `Project.toml`).
The examples will not be run on CI, but their rendered `CommonMark` outpout should be included
in the list of examples in the documentation of this package.

## Code style

We try to follow the [documentation guidelines](https://docs.julialang.org/en/v1/manual/documentation/)
from the Julia documentation as well as [Blue Style](https://github.com/invenia/BlueStyle).
We run [`JuliaFormatter.jl`](https://github.com/domluna/JuliaFormatter.jl) on the repo in
the way set in the `.JuliaFormatter.toml` file, which enforces a number of conventions consistent with the Blue Style.

We also follow a few internal conventions:

- Any implemented function should be accompanied by its mathematical formulae if a closed
  form exists.
- within a file the structs should come first and functions second. The only exception
  are constructors for the structs
- within both blocks an alphabetical order is preferable.
- The above implies that the mutating variant of a function follows the non-mutating variant.
- There should be no dangling `=` signs.
- Always add a newline between things of different types (struct/method/const).
- Always add a newline between methods for different functions (including mutating/nonmutating variants).
- Prefer to have no newline between methods for the same function; when reasonable,
  merge the docstrings into a generic function signature.
- All `import`/`using`/`include` should be in the main module file.
- There should only be a minimum of `export`s within this file, all problems should usually
  be later addressed as `ManoptExamples.[...]`
- the Quarto Markdown files are excluded from this formatting.
