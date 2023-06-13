#!/usr/bin/env julia
#
#

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
        Pkg.resolve()
        Pkg.instantiate()
        Pkg.build("IJulia") # build IJulia to the right version.
        Pkg.activate(@__DIR__) # but return to the docs one before
        run(`quarto render $(examples_folder)`)
    end
end

# (c) load necessary packages for the docs
using Documenter: DocMeta, HTML, MathJax3, deploydocs, makedocs
using ManoptExamples

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/JuliaManifolds/ManoptExamples.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)
open(joinpath(generated_path, "contributing.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)CONTRIBUTING.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
        println(io, line)
    end
end

makedocs(;
    format=HTML(; mathengine=MathJax3(), prettyurls=get(ENV, "CI", nothing) == "true"),
    #modules=[ManoptExamples],
    sitename="ManoptExamples.jl",
    modules=[ManoptExamples],
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Overview" => "examples/index.md",
            "Difference of Convex" => [
                "A Benchmark" => "examples/Difference-of-Convex-Benchmark.md",
                "Rosenbrock Metric" => "examples/Difference-of-Convex-Rosenbrock.md",
                "Frank Wolfe comparison" => "examples/Difference-of-Convex-Frank-Wolfe.md",
            ],
            "Riemannian Mean" => "examples/Riemannian-mean.md",
            "Robust PCA" => "examples/Robust-PCA.md",
            "Rosenbrock" => "examples/Rosenbrock.md",
        ],
        "Objectives" => "objectives/index.md",
        "Contributing to ManoptExamples.jl" => "contributing.md",
    ],
)
deploydocs(; repo="github.com/JuliaManifolds/ManoptExamples.jl", push_preview=true)
