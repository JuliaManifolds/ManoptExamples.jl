@doc raw"""
    robust_PCA_objective(data::AbstractMatrix, ε=1.0; evaluation=AllocatingEvaluation())
    robust_PCA_objective(M, data::AbstractMatrix, ε=1.0; evaluation=AllocatingEvaluton())

Generate the objective for the robust PCA task for some given `data` ``D`` and Huber regularization
parameter ``ε``.


# See also
[`RobustPCACost`](@ref ManoptExamples.RobustPCACost), [`RobustPCAGrad!!`](@ref ManoptExamples.RobustPCAGrad!!)

!!! note
    Since the construction is independent of the manifold, that argument is optional and
    mainly provided to comply with other objectives. Similarly, independent of the `evaluation`,
    indeed the gradient always allows for both the allocating and the in-place variant to be used,
    though that keyword is used to setup the objective.
"""
function robust_PCA_objective(
    data::AbstractMatrix, ε=1.0; evaluation=Manopt.AllocatingEvaluation()
)
    return Manopt.ManifoldGradientObjective(
        RobustPCACost(data, ε), RobustPCAGrad!!(data, 1.0; evaluation=evaluation)
    )
end
function robust_PCA_objective(
    M::AbstractManifold,
    data::AbstractMatrix,
    ε=1.0;
    evaluation=Manopt.AllocatingEvaluation(),
)
    return Manopt.ManifoldGradientObjective(
        RobustPCACost(M, data, ε),
        RobustPCAGrad!!(M, data, ε; evaluation=evaluation);
        evaluation=evaluation,
    )
end

@doc raw"""
    Riemannian_mean_objective(data, initial_vector=nothing, evaluation=AllocatingEvaluation())
    Riemannian_mean_objective(M, data;
    initial_vector=zero_vector(M, first(data)),
    evaluation=AllocatingEvaluton()
    )

Generate the objective for the Riemannian mean task for some given vector of
`data` points on the Riemannian manifold `M`.

# See also
[`RiemannianMeanCost`](@ref ManoptExamples.RiemannianMeanCost), [`RiemannianMeanGradient!!`](@ref ManoptExamples.RiemannianMeanGradient!!)

!!! note
    The first constructor should only be used if an additional storage of the vector is not
    feasible, since initialising the `initial_vector` to `nothing` disables the in-place variant.
    Hence the evaluation is a positional argument, since it only can be changed,
    if a vector is provided.
"""
function Riemannian_mean_objective(
    data::AbstractVector; initial_vector=nothing, evaluation=Manopt.AllocatingEvaluation()
)
    return Manopt.ManifoldGradientObjective(
        RiemannianMeanCost(data),
        RiemannianMeanGradient!!(data, initial_vector);
        evaluation=evaluation,
    )
end
function Riemannian_mean_objective(
    M::AbstractManifold,
    data;
    initial_vector=zero_vector(M, first(data)),
    evaluation=Manopt.AllocatingEvaluation(),
)
    return Manopt.ManifoldGradientObjective(
        RiemannianMeanCost(data),
        RiemannianMeanGradient!!(M, data; initial_vector=initial_vector);
        evaluation=evaluation,
    )
end