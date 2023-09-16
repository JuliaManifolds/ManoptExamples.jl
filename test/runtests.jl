using Manifolds, Manopt, ManoptExamples, Test

@testset "ManoptExamples.jl" begin
    include("test_rayleigh.jl")
    include("test_riemannian_mean.jl")
    include("test_robust_pca.jl")
    include("test_Rosenbrock.jl")
end
