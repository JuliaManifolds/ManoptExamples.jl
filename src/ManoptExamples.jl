module ManoptExamples
using Manopt, Manifolds, ManifoldDiff
# Common ollection of functions useful for several problems
include("functions/gradients.jl")

# Problems
include("problems/RiemannianMean.jl")
end
