The Riemannian Center of Mass (mean)
================
Ronny Bergmann
7/2/23

## Preliminary Notes

Each of the example objectives or problems stated in this package should
be accompanied by a [Quarto](https://quarto.org) notebook that illustrates their usage, like this one.

For this first example, the objective is a very common one, for example also used in the [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!/) tutorial of [Manopt.jl](https://manoptjl.org/).

The second goal of this tutorial is to also illustrate how this package provides these examples, namely in both an easy-to-use and a performant way.

There are two recommended ways to activate a reproducible environment.
For most cases the recommended environment is the one in `examples/`.
If you are programming a new, relatively short example, consider using the packages main environment, which is the same as having `ManoptExamples.jl` in development mode. this requires that your example does not have any (additional) dependencies beyond the ones `ManoptExamples.jl` has anyways.

For registered versions of `ManoptExamples.jl` use the environment of `examples/` and – under development – add `ManoptExamples.jl` in development mode from the parent folder. This should be changed after a new example is within a registered version to just use the `examples/` environment again.

``` julia
using Pkg;
Pkg.activate("."); # use the example environment,
```

## Loading packages and defining data

Loading the necessary packages and defining a data set on a manifold

``` julia
using ManoptExamples, Manopt, Manifolds, ManifoldDiff, Random
Random.seed!(42)
M = Sphere(2)
n = 100
σ = π / 8
p = 1 / sqrt(2) * [1.0, 0.0, 1.0]
data = [exp(M, p,  σ * rand(M; vector_at=p)) for i in 1:n];
```

## Variant 1: Using the functions

We can define both the cost and gradient, [`RiemannianMeanCost`](@ref ManoptExamples.RiemannianMeanCost) and [`RiemannianMeanGradient!!`](@ref ManoptExamples.RiemannianMeanGradient!!), respectively.
For their mathematical derivation and further explanations,
we again refer to [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!/).

``` julia
f = ManoptExamples.RiemannianMeanCost(data)
grad_f = ManoptExamples.RiemannianMeanGradient!!(M, data)
```

Then we can for example directly call a [gradient descent](https://manoptjl.org/stable/solvers/gradient_descent/) as

``` julia
x1 = gradient_descent(M, f, grad_f, first(data))
```

    3-element Vector{Float64}:
     0.6868392794764022
     0.006531600682543096
     0.726779982085954

## Variant 2: Using the objective

A shorter way to directly obtain the [Manifold objective](https://manoptjl.org/stable/plans/objective/) including these two functions.
Here, we want to specify that the objective can do inplace-evaluations using the `evaluation=`-keyword. The objective can be obtained calling
[`Riemannian_mean_objective`](@ref ManoptExamples.Riemannian_mean_objective) as

``` julia
rmo = ManoptExamples.Riemannian_mean_objective(
    M, data,
    evaluation=InplaceEvaluation(),
)
```

Together with a manifold, this forms a [Manopt Problem](https://manoptjl.org/stable/plans/problem/), which would usually enable to switch manifolds between solver runs. Here we could for example switch to using `Euclidean(3)` instead for the same data the objective is build upon.

``` julia
rmp = DefaultManoptProblem(M, rmo)
```

This enables us to for example solve the task with different, gradient based, solvers. The first is the same as above, just not using the high-level interface

``` julia
s1 = GradientDescentState(M, copy(M, first(data)))
solve!(rmp, s1)
x2 = get_solver_result(s1)
```

    3-element Vector{Float64}:
     0.6868395649618767
     0.006531393870513675
     0.7267797141480264

but we can easily use a conjugate gradient instead

``` julia
s2 = ConjugateGradientDescentState(
    M,
    copy(M, first(data)),
    StopAfterIteration(100),
    ArmijoLinesearch(M),
    FletcherReevesCoefficient(),
)
solve!(rmp, s2)
x3 = get_solver_result(s2)
```

    3-element Vector{Float64}:
     0.6868393265070905
     0.006531566700408201
     0.7267799379452656
