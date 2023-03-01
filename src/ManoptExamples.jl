module ManoptExamples
using ManifoldsBase, Manopt, Manifolds, ManifoldDiff
# Common ollection of functions useful for several problems

# Objetives
include("objectives/RiemannianMean.jl")
include("objectives/RobustPCA.jl")
include("objectives/Rosenbrock.jl")
end
