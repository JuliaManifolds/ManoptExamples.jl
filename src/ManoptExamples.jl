module ManoptExamples
using Manopt, Manifolds, ManifoldDiff
# Common ollection of functions useful for several problems
include("functions/gradients.jl")

# Objetives
include("objectives/RiemannianMean.jl")
end
