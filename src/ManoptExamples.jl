@doc raw"""
🏔️⛷️ ManoptExamples.jl – A collection of research and tutorial example problems for [Manopt.jl](https://manoptjl.org)

* 📚 Documentation: [juliamanifolds.github.io/ManoptExamples.jl](https://juliamanifolds.github.io/ManoptExamples.jl/stable)
* 📦 Repository: [github.com/JuliaManifolds/ManoptExamples.jl](https://github.com/JuliaManifolds/ManoptExamples.jl)
* 💬 Discussions: [github.com/JuliaManifolds/ManoptExamples.jl/discussions](https://github.com/JuliaManifolds/ManoptExamples.jl/discussions)
* 🎯 Issues: [github.com/JuliaManifolds/ManoptExamples.jl/issues](https://github.com/JuliaManifolds/ManoptExamples.jl/issues)
"""
module ManoptExamples
using LinearAlgebra: dot, Symmetric, Diagonal, I
using ManifoldsBase, Manifolds, ManifoldDiff
using ManifoldsBase: TypeParameter
using OffsetArrays
using SparseArrays
import ManifoldsBase: exp!, exp, inner, log, log!
import Manifolds:
    change_representer, change_representer!, local_metric, inverse_local_metric
import Manifolds: Euclidean, Circle, PositiveNumbers
import Manifolds: Sphere, SymmetricPositiveDefinite
using Markdown: @doc_str
using ManifoldDiff:
    adjoint_differential_log_basepoint,
    adjoint_differential_log_basepoint!,
    adjoint_differential_log_argument,
    adjoint_differential_log_argument!,
    adjoint_differential_shortest_geodesic_startpoint,
    adjoint_differential_shortest_geodesic_startpoint!,
    adjoint_differential_shortest_geodesic_endpoint,
    adjoint_differential_shortest_geodesic_endpoint!,
    differential_log_argument,
    differential_log_argument!,
    differential_log_basepoint,
    differential_log_basepoint!,
    differential_shortest_geodesic_startpoint,
    differential_shortest_geodesic_startpoint!,
    differential_shortest_geodesic_endpoint,
    differential_shortest_geodesic_endpoint!,
    grad_distance
using Requires

const NONMUTATINGMANIFOLDS = Union{Circle,PositiveNumbers,Euclidean{Tuple{}}}

function __init__()
    #
    # Requires fallback for Julia < 1.9
    #
    @static if !isdefined(Base, :get_extension)
        @require Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5" begin
            include("../ext/ManoptExamplesManoptExt.jl")
        end
    end
end

# Common ollection of functions useful for several problems

# Objetives
include("objectives/BezierCurves.jl")
include("objectives/RayleighQuotient.jl")
include("objectives/RiemannianMean.jl")
include("objectives/RobustPCA.jl")
include("objectives/Rosenbrock.jl")
include("objectives/TotalVariation.jl")
include("objectives/VariationalProblemAssembler.jl")

include("data/artificial_signals.jl")
include("data/artificial_images.jl")

include("ErrorMeasures.jl")

export exp!, exp, inner, log, log!
export change_representer, change_representer!, local_metric, inverse_local_metric
end
