@doc raw"""
🏔️⛷️ ManoptExamples.jl – A collection of research and tutorial example problems for [Manopt.jl](https://manoptjl.org)

* 📚 Documentation: [juliamanifolds.github.io/ManoptExamples.jl](https://juliamanifolds.github.io/ManoptExamples.jl/stable)
* 📦 Repository: [github.com/JuliaManifolds/ManoptExamples.jl](https://github.com/JuliaManifolds/ManoptExamples.jl)
* 💬 Discussions: [github.com/JuliaManifolds/ManoptExamples.jl/discussions](https://github.com/JuliaManifolds/ManoptExamples.jl/discussions)
* 🎯 Issues: [github.com/JuliaManifolds/ManoptExamples.jl/issues](https://github.com/JuliaManifolds/ManoptExamples.jl/issues)
"""
module ManoptExamples
using ManifoldsBase, Manopt, Manifolds, ManifoldDiff
import ManifoldsBase: exp!, exp, inner, log, log!
import Manifolds:
    change_representer, change_representer!, local_metric, inverse_local_metric
# Common ollection of functions useful for several problems

# Objetives
include("objectives/RiemannianMean.jl")
include("objectives/RobustPCA.jl")
include("objectives/Rosenbrock.jl")

export exp!, exp, inner, log, log!
export change_representer, change_representer!, local_metric, inverse_local_metric
end
