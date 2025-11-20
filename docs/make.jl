#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
            docs/make.jl

        Render the `ManoptExamples.jl` documenation.

        Arguments
        * `--exclude-examples` - exclude the examples from the menu of Documenter,
          this can be used if you do not have Quarto installed to still be able to render the docs
          locally on this machine. This option should not be set on CI.
        * `--help`         - print this help and exit without rendering the documentation
        * `--prettyurls`   – toggle the pretty urls part to true (which is otherwise only true on CI)
        * `--quarto`       – run the Quarto notebooks from the `examples/` folder before generating the documentation
          this has to be run locally at least once for the `examples/*.md` files to exist that are included in
          the documentation (see `--exclude-examples`) for the alternative.
          If they are generated ones they are cached accordingly.
          Then you can spare time in the rendering by not passing this argument.
        """,
    )
    exit(0)
end

run_quarto = "--quarto" in ARGS
run_on_CI = (get(ENV, "CI", nothing) == "true")
tutorials_in_menu = !("--exclude-tutorials" ∈ ARGS)
#
# (a) if docs is not the current active environment, switch to it
# (from https://github.com/JuliaIO/HDF5.jl/pull/1020/) 
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# (b) If quarto is set, or we are on CI, run quarto
if run_quarto || run_on_CI
    @info "Rendering Quarto"
    example_folder = (@__DIR__) * "/../examples"
    # instantiate the examples environment if necessary
    Pkg.activate(example_folder)
    # For a breaking release -> also set the examples folder to the most recent version
    Pkg.instantiate()
    Pkg.activate(@__DIR__) # but return to the docs one before
    run(`quarto render $(example_folder)`)
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

# (d) add contributing.md and Changelog.md to docs

function add_links(line::String, url::String = "https://github.com/JuliaManifolds/Manifolds.jl")
    # replace issues (#XXXX) -> ([#XXXX](url/issue/XXXX))
    while (m = match(r"\(\#([0-9]+)\)", line)) !== nothing
        id = m.captures[1]
        line = replace(line, m.match => "([#$id]($url/issues/$id))")
    end
    # replace ## [X.Y.Z] -> with a link to the release [X.Y.Z](url/releases/tag/vX.Y.Z)
    while (m = match(r"\#\# \[([0-9]+.[0-9]+.[0-9]+)\] (.*)", line)) !== nothing
        tag = m.captures[1]
        date = m.captures[2]
        line = replace(line, m.match => "## [$tag]($url/releases/tag/v$tag) ($date)")
    end
    return line
end

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
            println(io, add_links(line))
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
    "LTMADS" => [
        "Spectral & Robust Procrustes" => "examples/Spectral-Procrustes-2D.md",
    ],
    "Projected Gradient Algorithm" => [
        raw"Mean on $\mathbb H^2$" => "examples/Constrained-Mean-H2.md",
        raw"Mean on $\mathbb H^n$" => "examples/Constrained-Mean-Hn.md",
    ],
    "Hyperparameter optimziation" => "examples/HyperparameterOptimization.md",
    "The Rayleigh Quotient" => "examples/RayleighQuotient.md",
    "Riemannian Mean" => "examples/Riemannian-mean.md",
    "Proximal Gradient Methods" => [
        "Sparse PCA" => "examples/NCRPG-Sparse-PCA.md",
        "Grassmann Experiment" => "examples/NCRPG-Grassmann.md",
        "Row-Sparse Low-Rank Matrix Recovery" => "examples/NCRPG-Row-Sparse-Low-Rank.md",
        "Convex Example on SPDs" => "examples/CRPG-Convex-SPD.md",
        raw"Sparse Approximation on $\mathbb H^n$" => "examples/CRPG-Sparse-Approximation.md",
        raw"Mean on $\mathbb H^n$" => "examples/CRPG-Constrained-Mean-Hn.md",
    ],
    "Robust PCA" => "examples/Robust-PCA.md",
    "Rosenbrock" => "examples/Rosenbrock.md",
    "Total Variation" => "examples/Total-Variation.md",
]
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :alpha)
links = InterLinks(
    "ManifoldDiff" => ("https://juliamanifolds.github.io/ManifoldDiff.jl/stable/"),
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
    "Manifolds" => ("https://juliamanifolds.github.io/Manifolds.jl/stable/"),
    "Manopt" => ("https://juliamanifolds.github.io/Manopt.jl/stable/"),
)
# (e) ...finally! make docs
makedocs(;
    format = Documenter.HTML(;
        prettyurls = (get(ENV, "CI", nothing) == "true") || ("--prettyurls" ∈ ARGS),
        assets = ["assets/favicon.ico", "assets/citations.css"],
        size_threshold_warn = 200 * 2^10, # raise slightly to 200 KiB
        size_threshold = 300 * 2^10,      # raise slightly to 300 KiB
    ),
    authors = "Ronny Bergmann, Hajg Jasa",
    sitename = "ManoptExamples.jl",
    modules = [ManoptExamples],
    pages = [
        "Home" => "index.md",
        (examples_in_menu ? [examples_menu] : [])...,
        "Objectives" => "objectives/index.md",
        "Data" => "data/index.md",
        "Contributing to ManoptExamples.jl" => "contributing.md",
        "Changelog" => "changelog.md",
        "References" => "references.md",
    ],
    plugins = [bib, links],
)
deploydocs(; repo = "github.com/JuliaManifolds/ManoptExamples.jl", push_preview = true)
