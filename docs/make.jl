#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
    docs/make.jl

Render the `ManoptExamples.jl` documenation with optional arguments

Arguments
* `--exclude-examples` - exclude the examples from the menu of Documenter,
  this can be used if you do not have Quarto installed to still be able to render the docs
  locally on this machine. This option should not be set on CI.
* `--help`         - print this help and exit without rendering the documentation
* `--prettyurls`   – toggle the pretty urls part to true (which is otherwise only true on CI)
* `--quarto`       – run the Quarto notebooks from the `tutorials/` folder before generating the documentation
  this has to be run locally at least once for the `tutorials/*.md` files to exist that are included in
  the documentation (see `--exclude-tutorials`) for the alternative.
  If they are generated ones they are cached accordingly.
  Then you can spare time in the rendering by not passing this argument.
""",
    )
    exit(0)
end

#
# (a) if docs is not the current active environment, switch to it
# (from https://github.com/JuliaIO/HDF5.jl/pull/1020/) 
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

# (b) Did someone say render? Then we render!
if "--quarto" ∈ ARGS
    using CondaPkg
    CondaPkg.withenv() do
        @info "Rendering Quarto"
        examples_folder = (@__DIR__) * "/../examples"
        # instantiate the tutorials environment if necessary
        Pkg.activate(examples_folder)
        Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../")) # Before resolving set ManoptExamples to dev
        Pkg.resolve()
        Pkg.instantiate()
        Pkg.build("IJulia") # build IJulia to the right version.
        Pkg.activate(@__DIR__) # but return to the docs one before
        CondaPkg.add("optuna")
        run(`quarto render $(examples_folder)`)
    end
end

examples_in_menu = true
if "--exclude-examples" ∈ ARGS
    @warn """
    You are excluding the examples from the Menu,
    which might be done if you can not render them locally.

    Remember that this should never be done on CI for the full documentation.
    """
    examples_in_menu = false
end

# (c) load necessary packages for the docs
using Documenter
using ManoptExamples
using DocumenterInterLinks
using DocumenterCitations

# (d) add contributing.md to docs
generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/JuliaManifolds/Manopt.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)
for (md_file, doc_file) in
    [("CONTRIBUTING.md", "contributing.md"), ("Changelog.md", "changelog.md")]
    open(joinpath(generated_path, doc_file), "w") do io
        # Point to source license file
        println(
            io,
            """
            ```@meta
            EditURL = "$(base_url)$(md_file)"
            ```
            """,
        )
        # Write the contents out below the meta block
        for line in eachline(joinpath(dirname(@__DIR__), md_file))
            println(io, line)
        end
    end
end

## Build examples menu
examples_menu =
    "Examples" => [
        "Overview" => "examples/index.md",
        "Difference of Convex" => [
            "A Benchmark" => "examples/Difference-of-Convex-Benchmark.md",
            "Rosenbrock Metric" => "examples/Difference-of-Convex-Rosenbrock.md",
            "Frank Wolfe comparison" => "examples/Difference-of-Convex-Frank-Wolfe.md",
        ],
        "Convex Bundle Method" => [
            "Riemannian Median" => "examples/RCBM-Median.md",
            "Hyperbolic Signal Denoising" => "examples/H2-Signal-TV.md",
            "Spectral Procrustes" => "examples/Spectral-Procrustes.md",
        ],
        "Projected Gradient Algorithm" => [
            raw"Mean on $\mathbb H^2$" => "examples/Constrained-Mean-H2.md",
            raw"Mean on $\mathbb H^n$" => "examples/Constrained-Mean-Hn.md",
        ],
        "Hyperparameter optimziation" => "examples/HyperparameterOptimization.md",
        "The Rayleigh Quotient" => "examples/RayleighQuotient.md",
        "Riemannian Mean" => "examples/Riemannian-mean.md",
        "Robust PCA" => "examples/Robust-PCA.md",
        "Rosenbrock" => "examples/Rosenbrock.md",
        "Total Variation" => "examples/Total-Variation.md",
    ]
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:alpha)
links = InterLinks(
    "ManifoldDiff" => ("https://juliamanifolds.github.io/ManifoldDiff.jl/stable/"),
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
    "Manifolds" => ("https://juliamanifolds.github.io/Manifolds.jl/stable/"),
    "Manopt" => ("https://juliamanifolds.github.io/Manopt.jl/stable/"),
)
# (e) ...finally! make docs
makedocs(;
    format=Documenter.HTML(;
        prettyurls=(get(ENV, "CI", nothing) == "true") || ("--prettyurls" ∈ ARGS),
        assets=["assets/favicon.ico", "assets/citations.css"],
        size_threshold_warn=200 * 2^10, # raise slightly to 200 KiB
        size_threshold=300 * 2^10,      # raise slightly to 300 KiB
    ),
    authors="Ronny Bergmann",
    sitename="ManoptExamples.jl",
    modules=[ManoptExamples],
    pages=[
        "Home" => "index.md",
        (examples_in_menu ? [examples_menu] : [])...,
        "Objectives" => "objectives/index.md",
        "Data" => "data/index.md",
        "Contributing to ManoptExamples.jl" => "contributing.md",
        "Changelog" => "changelog.md",
        "References" => "references.md",
    ],
    plugins=[bib,links],
)
deploydocs(; repo="github.com/JuliaManifolds/ManoptExamples.jl", push_preview=true)
