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
            "Riemannian Mean" => "examples/Riemannian-mean.md",
            ],
        "Objectives" => "objectives/index.md",
        "Functions" => ["Gradients" => "functions/gradients.md"],
        "Contributing to ManoptExamples.jl" => "contributing.md",
    ],
)
deploydocs(; repo="github.com/JuliaManifolds/ManoptExamples.jl", push_preview=true)
