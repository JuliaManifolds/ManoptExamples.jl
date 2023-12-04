using Manifolds, Manopt, ManoptExamples, Test

@testset "ManoptExamples.jl" begin
    include("test_bezier.jl")
    #    include("test_data.jl")
    #    include("test_error_measures.jl")
    include("test_rayleigh.jl")
    include("test_riemannian_mean.jl")
    include("test_robust_pca.jl")
    include("test_Rosenbrock.jl")
    include("test_total_variation.jl")
end
