module ManoptExamples
using Manopt, Manifolds, ManifoldDiff
# Common ollection of functions useful for several problems

# Objetives
include("objectives/RiemannianMean.jl")
include("objectives/RobustPCA.jl")
end
