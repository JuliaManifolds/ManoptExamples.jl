module ManoptExamplesManoptExt

if isdefined(Base, :get_extension)
    using Manopt
    using ManoptExamples
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..ManoptExamples
    using ..Manopt
end

using ManifoldsBase
#
#
# Objectives
function ManoptExamples.robust_PCA_objective(
    data::AbstractMatrix, ε=1.0; evaluation=Manopt.AllocatingEvaluation()
)
    return Manopt.ManifoldGradientObjective(
        ManoptExamples.RobustPCACost(data, ε),
        ManoptExamples.RobustPCAGrad!!(data, 1.0; evaluation=evaluation),
    )
end
function ManoptExamples.robust_PCA_objective(
    M::AbstractManifold,
    data::AbstractMatrix,
    ε=1.0;
    evaluation=Manopt.AllocatingEvaluation(),
)
    return Manopt.ManifoldGradientObjective(
        ManoptExamples.RobustPCACost(M, data, ε),
        ManoptExamples.RobustPCAGrad!!(M, data, ε; evaluation=evaluation);
        evaluation=evaluation,
    )
end

function ManoptExamples.Riemannian_mean_objective(
    data::AbstractVector; initial_vector=nothing, evaluation=Manopt.AllocatingEvaluation()
)
    return Manopt.ManifoldGradientObjective(
        ManoptExamples.RiemannianMeanCost(data),
        ManoptExamples.RiemannianMeanGradient!!(data, initial_vector);
        evaluation=evaluation,
    )
end
function ManoptExamples.Riemannian_mean_objective(
    M::AbstractManifold,
    data;
    initial_vector=zero_vector(M, first(data)),
    evaluation=Manopt.AllocatingEvaluation(),
)
    return Manopt.ManifoldGradientObjective(
        ManoptExamples.RiemannianMeanCost(data),
        ManoptExamples.RiemannianMeanGradient!!(M, data; initial_vector=initial_vector);
        evaluation=evaluation,
    )
end

function ManoptExamples.Rosenbrock_objective(
    M::AbstractManifold=ManifoldsBase.DefaultManifold();
    a=100.0,
    b=1.0,
    evaluation=Manopt.AllocatingEvaluation(),
)
    return Manopt.ManifoldGradientObjective(
        ManoptExamples.RosenbrockCost(M; a=a, b=b),
        ManoptExamples.RosenbrockGradient!!(M; a=a, b=b, evaluation=evaluation);
        evaluation=evaluation,
    )
end

#
#
# prox with a subsolver
#
function ManoptExamples.prox_second_order_Total_Variation(
    M::AbstractManifold,
    λ,
    x::Tuple{T,T,T},
    p::Int=1;
    stopping_criterion::StoppingCriterion=Manopt.StopAfterIteration(10),
    kwargs...,
) where {T}
    (p != 1) && throw(
        ErrorException(
            "Proximal Map of TV2(M, λ, x, p) not implemented for p=$(p) (requires p=1) on general manifolds.",
        ),
    )
    PowX = [x...]
    PowM = ManifoldsBase.PowerManifold(M, ManifoldsBase.NestedPowerRepresentation(), 3)
    xR = PowX
    function F(M, x)
        return 1 / 2 * distance(M, PowX, x)^2 +
               λ * ManoptExamples.second_order_Total_Variation(M, x)
    end
    function ∂F(PowM, x)
        return log(PowM, x, PowX) +
               λ * ManoptExamples.grad_second_order_Total_Variation(PowM, x)
    end
    Manopt.subgradient_method!(
        PowM, F, ∂F, xR; stopping_criterion=stopping_criterion, kwargs...
    )
    return (xR...,)
end
function ManoptExamples.prox_second_order_Total_Variation!(
    M::AbstractManifold,
    y,
    λ,
    x::Tuple{T,T,T},
    p::Int=1;
    stopping_criterion::StoppingCriterion=Manopt.StopAfterIteration(10),
    kwargs...,
) where {T}
    (p != 1) && throw(
        ErrorException(
            "Proximal Map of TV2(M, λ, x, p) not implemented for p=$(p) (requires p=1) on general manifolds.",
        ),
    )
    PowX = [x...]
    PowM = PowerManifold(M, NestedPowerRepresentation(), 3)
    copyto!(M, y, PowX)
    function F(M, x)
        return 1 / 2 * distance(M, PowX, x)^2 +
               λ * ManoptExamples.second_order_Total_Variation(M, x)
    end
    function ∂F!(M, y, x)
        return log!(M, y, x, PowX) +
               λ * ManoptExamples.grad_second_order_Total_Variation!(M, y, x)
    end
    Manopt.subgradient_method!(
        PowM,
        F,
        ∂F!,
        y;
        stopping_criterion=stopping_criterion,
        evaluation=Manopt.InplaceEvaluation(),
        kwargs...,
    )
    return y
end

end
