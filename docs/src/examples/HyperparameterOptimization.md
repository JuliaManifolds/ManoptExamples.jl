# Hyperparameter optimization
Mateusz Baran
2024-08-03

## Introduction

This example shows how to automatically select the best values of hyperparameters of optimization procedures such as retraction, vector transport, size of memory in L-BFGS or line search coefficients.
Hyperparameter optimization relies on the [Optuna](https://optuna.org/) [AkibaSanoYanaseToshihikoTakeruKoyama:2019](@cite) Python library because it is much more advanced than similar Julia projects, offering Bayesian optimization with conditional hyperparameters and early stopping.

## General definitions

Here are some general definitions that you will most likely be able to directly use for your problem without any changes.
Just remember to install `optuna`, for example using `CondaPkg` Julia library.

``` julia
using Manifolds, Manopt
using PythonCall
using BenchmarkTools
using LineSearches

# This script requires optuna to be available through PythonCall
# You can install it for example using
# using CondaPkg
# ]conda add optuna

optuna = pyimport("optuna")

norm_inf(M::AbstractManifold, p, X) = norm(X, Inf)

struct TTsuggest_int
    suggestions::Dict{String,Int}
end
function (s::TTsuggest_int)(name::String, a, b)
    return s.suggestions[name]
end
struct TTsuggest_float
    suggestions::Dict{String,Float64}
end
function (s::TTsuggest_float)(name::String, a, b; log::Bool=false)
    return s.suggestions[name]
end
struct TTsuggest_categorical
    suggestions::Dict{String,Any}
end
function (s::TTsuggest_categorical)(name::String, vals)
    return s.suggestions[name]
end
struct TTreport
    reported_vals::Vector{Float64}
end
function (r::TTreport)(val, i)
    return push!(r.reported_vals, val)
end
struct TTshould_prune end
(::TTshould_prune)() = Py(false)
struct TracingTrial
    suggest_int::TTsuggest_int
    suggest_float::TTsuggest_float
    suggest_categorical::TTsuggest_categorical
    report::TTreport
    should_prune::TTshould_prune
end
```

        CondaPkg Found dependencies: /home/mateusz/.julia/packages/PythonCall/wXfah/CondaPkg.toml
        CondaPkg Found dependencies: /home/mateusz/.julia/environments/v1.10/CondaPkg.toml
        CondaPkg Dependencies already up to date

The next part is your hyperparameter optimization objective. The `ObjectiveData` struct contains all relevant information about the sequence of specific problems.
The outermost key part is the `N_range` field.
Early stopping requires a series of progressively more complex problems.
They will be attempted from the most simple one to the most complex one, and are specified by the values of `N` in that vector.

``` julia
mutable struct ObjectiveData{TObj,TGrad}
    obj::TObj
    grad::TGrad
    N_range::Vector{Int}
    gtol::Float64
    vts::Vector{AbstractVectorTransportMethod}
    retrs::Vector{AbstractRetractionMethod}
    manifold_constructors::Vector{Tuple{String,Any}}
    pruning_losses::Vector{Float64}
    manopt_stepsize::Vector{Tuple{String,Any}}
    obj_loss_coeff::Float64
end


function compute_pruning_losses(
    od::ObjectiveData,
    int_suggestions::Dict{String,Int},
    float_suggestions::Dict{String,Float64},
    categorical_suggestions::Dict{String,Int},
)
    tt = TracingTrial(
        TTsuggest_int(int_suggestions),
        TTsuggest_float(float_suggestions),
        TTsuggest_categorical(categorical_suggestions),
        TTreport(Float64[]),
        TTshould_prune(),
    )
    od(tt)
    return tt.report.reported_vals
end
```

    compute_pruning_losses (generic function with 1 method)

In the example below we optimize hyperparameters on a sequence of Rosenbrock-type problems restricted to spheres:

``` math
\arg\min_{p \in S^{N-1}} \sum_{i=1}^{N/2} (1-p_{2i})^2 + 100 (p_{2i+1} - p_{2i}^2)^2,
```

where $N \in [2, 16, 128, 1024, 8192, 65536]$.

`obj` and `grad` are the objective and gradient, here defined as below.
Note that gradient works in-place and variants without manifolds are also provided for easier comparison with other libraries like `Optim.jl`.
It is easiest when problems for different values `N` can be distinguished by being defined on successively larger manifolds but the script could be modified so that it’s not necessary.

`pruning_losses` and `compute_pruning_losses` are related to early pruning used in Optuna and you shouldn’t have to modify them.

``` julia
function f_rosenbrock(x)
    result = 0.0
    for i in 1:2:length(x)
        result += (1.0 - x[i])^2 + 100.0 * (x[i + 1] - x[i]^2)^2
    end
    return result
end
function f_rosenbrock(::AbstractManifold, x)
    return f_rosenbrock(x)
end

function g_rosenbrock!(storage, x)
    for i in 1:2:length(x)
        storage[i] = -2.0 * (1.0 - x[i]) - 400.0 * (x[i + 1] - x[i]^2) * x[i]
        storage[i + 1] = 200.0 * (x[i + 1] - x[i]^2)
    end
    return storage
end
function g_rosenbrock!(M::AbstractManifold, storage, x)
    g_rosenbrock!(storage, x)
    riemannian_gradient!(M, storage, x, storage)
    return storage
end
```

    g_rosenbrock! (generic function with 2 methods)

Next, `gtol` is the tolerance used for the stopping criterion in optimization.
`vts` and `retrs` are, respectively, vector transports and retraction methods selected through hyperparameter optimization.
Some items need to be different for different values of `N`, for example the manifold over which the problem is defined.
This is handled by `manifold_constructors` which is then defined as `Tuple{String,Any}[("Sphere", N -> Manifolds.Sphere(N - 1))]`, where the string `"Sphere"` is used to identify the manifold family and the next element is a function that transforms the value of `N` to the manifold for the problem of size `N`.

Similarly, different stepsize selection methods may be considered.
This is handled by the field `manopt_stepsize`.
It will be easiest to see how it works by looking at how it is initialized:

    Tuple{String,Any}[
        ("LS-HZ", M -> Manopt.LineSearchesStepsize(ls_hz)),
        ("Wolfe-Powell", (M, c1, c2) -> Manopt.WolfePowellLinesearch(M, c1, c2)),
    ]

We have a string that identifies the line search method name and a constructor of the line search which takes relevant arguments like the manifold or a numerical parameter.

The next part is the trial evaluation procedure.
This is one of the more important places which need to be customized to your problem.
This is the point where we tell Optuna about the relevant optimization hyperparameters and use them to define specific problems.
The hyperparameter optimization is a multiobjective problem: we want as good problem objectives as possible and as low times as possible.
As Optuna doesn’t currently support multicriteria pruning, which is important for obtaining a solution in a reasonable amount of time, we use a linear combination of sub-objectives to turn the problem into a single-criterion optimization.
The hyperparameter optimization objective is a linear combination of achieved objectives the relative weight is controlled by `objective.obj_loss_coeff`.

``` julia
function (objective::ObjectiveData)(trial)
    # Here we use optuna to select memory length for L-BFGS -- an integer in the range between 2 and 30, referenced by name "mem_len"
    mem_len = trial.suggest_int("mem_len", 2, 30)

    # Here we select a vector transport and retraction methods, one of those specified in the `ObjectiveData`.
    vt = objective.vts[pyconvert(
        Int,
        trial.suggest_categorical(
            "vector_transport_method", Vector(eachindex(objective.vts))
        ),
    )]
    retr = objective.retrs[pyconvert(
        Int,
        trial.suggest_categorical("retraction_method", Vector(eachindex(objective.retrs))),
    )]

    # Here we select the manifold constructor, in case we want to try different manifolds for our problem. For example one could try defining a problem with orthogonality constraints on Stiefel, Grassmann or flag manifold.
    manifold_name, manifold_constructor = objective.manifold_constructors[pyconvert(
        Int,
        trial.suggest_categorical(
            "manifold", Vector(eachindex(objective.manifold_constructors))
        ),
    )]

    # Here the stepsize selection method type is selected.
    manopt_stepsize_name, manopt_stepsize_constructor = objective.manopt_stepsize[pyconvert(
        Int,
        trial.suggest_categorical(
            "manopt_stepsize", Vector(eachindex(objective.manopt_stepsize))
        ),
    )]

    # This parametrizes stepsize selection methods with relevant numerical parameters.
    local c1_val, c2_val, hz_sigma
    if manopt_stepsize_name == "Wolfe-Powell"
        c1_val = pyconvert(
            Float64, trial.suggest_float("Wolfe-Powell c1", 1e-5, 1e-2; log=true)
        )
        c2_val =
            1.0 - pyconvert(
                Float64, trial.suggest_float("Wolfe-Powell 1-c2", 1e-4, 1e-2; log=true)
            )
    elseif manopt_stepsize_name == "Improved HZ"
        hz_sigma = pyconvert(Float64, trial.suggest_float("Improved HZ sigma", 0.1, 0.9))
    end

    # The current loss estimate, taking into account estimated loss values for larger, not-yet-evaluated values of `N`.
    loss = sum(objective.pruning_losses)

    # Here iterate over problems we want to optimize for
    # from smallest to largest; pruning should stop the iteration early
    # if the hyperparameter set is not promising
    cur_i = 0
    for N in objective.N_range
        # Here we define the initial point for the optimization procedure
        p0 = zeros(N)
        p0[1] = 1
        M = manifold_constructor(N)
        # Here we construct the specific line search to be used
        local ls
        if manopt_stepsize_name == "Wolfe-Powell"
            ls = manopt_stepsize_constructor(M, c1_val, c2_val)
        elseif manopt_stepsize_name == "Improved HZ"
            ls = manopt_stepsize_constructor(M, hz_sigma)
        else
            ls = manopt_stepsize_constructor(M)
        end
        manopt_time, manopt_iters, manopt_obj = benchmark_time_state(
            ManoptQN(),
            M,
            N,
            objective.obj,
            objective.grad,
            p0,
            ls,
            pyconvert(Int, mem_len),
            objective.gtol;
            vector_transport_method=vt,
            retraction_method=retr,
        )
        # TODO: turn this into multi-criteria optimization when Optuna starts supporting
        # pruning in such problems
        loss -= objective.pruning_losses[cur_i + 1]
        loss += manopt_time + objective.obj_loss_coeff * manopt_obj
        trial.report(loss, cur_i)
        if pyconvert(Bool, trial.should_prune().__bool__())
            throw(PyException(optuna.TrialPruned()))
        end
        cur_i += 1
    end
    return loss
end
```

In the following benchmarking code you will most likely have to adapt solver parameters.
This is designed around `quasi_Newton` but can be adapted to any solver as needed.
The example below performs a small number of trials for quick rendering but in practice you should aim for at least a few thousand trials (the `n_trials` parameter).

``` julia
# An abstract type in case we want to try different optimization packages.
abstract type AbstractOptimConfig end
struct ManoptQN <: AbstractOptimConfig end

# Benchmark that evaluates hyperparameters. Returns time to reach the solution, number of iterations and final value of the objective.
function benchmark_time_state(
    ::ManoptQN,
    M::AbstractManifold,
    N,
    f,
    g!,
    p0,
    stepsize::Manopt.Stepsize,
    mem_len::Int,
    gtol::Real;
    kwargs...,
)
    manopt_sc = StopWhenGradientNormLess(gtol; norm=norm_inf) | StopAfterIteration(1000)
    mem_len = min(mem_len, manifold_dimension(M))
    manopt_state = quasi_Newton(
        M,
        f,
        g!,
        p0;
        stepsize=stepsize,
        evaluation=InplaceEvaluation(),
        return_state=true,
        memory_size=mem_len,
        stopping_criterion=manopt_sc,
        debug=[],
        kwargs...,
    )
    bench_manopt = @benchmark quasi_Newton(
        $M,
        $f,
        $g!,
        $p0;
        stepsize=$(stepsize),
        evaluation=$(InplaceEvaluation()),
        memory_size=$mem_len,
        stopping_criterion=$(manopt_sc),
        debug=[],
        $kwargs...,
    )
    iters = get_count(manopt_state, :Iterations)
    final_val = f(M, manopt_state.p)
    return median(bench_manopt.times) / 1000, iters, final_val
end

"""
    lbfgs_study(; pruning_coeff::Float64=0.95)

Set up the example hyperparameter optimization study.
"""
function lbfgs_study(; pruning_coeff::Float64=0.95)
    Ns = [2^n for n in 1:3:12]
    ls_hz = LineSearches.HagerZhang()
    od = ObjectiveData(
        f_rosenbrock,
        g_rosenbrock!,
        Ns,
        1e-5,
        AbstractVectorTransportMethod[ParallelTransport(), ProjectionTransport()],
        [ExponentialRetraction(), ProjectionRetraction()],
        Tuple{String,Any}[("Sphere", N -> Manifolds.Sphere(N - 1))],
        zeros(Float64, eachindex(Ns)),
        Tuple{String,Any}[
            ("LS-HZ", M -> Manopt.LineSearchesStepsize(ls_hz)),
            #("Improved HZ", (M, sigma) -> HagerZhangLinesearch(M; sigma=sigma)),
            ("Wolfe-Powell", (M, c1, c2) -> Manopt.WolfePowellLinesearch(M, c1, c2)),
        ],
        10.0,
    )

    # Here you need to define baseline values of all hyperparameters
    baseline_pruning_losses = compute_pruning_losses(
        od,
        Dict("mem_len" => 4),
        Dict(
            "Wolfe-Powell c1" => 1e-4,
            "Wolfe-Powell 1-c2" => 1e-3,
            "Improved HZ sigma" => 0.9,
        ),
        Dict(
            "vector_transport_method" => 1,
            "retraction_method" => 1,
            "manifold" => 1,
            "manopt_stepsize" => 1,
        ),
    )
    od.pruning_losses = pruning_coeff * baseline_pruning_losses

    study = optuna.create_study(; study_name="L-BFGS")
    # Here you can specify number of trials and timeout (in seconds).
    study.optimize(od; n_trials=1000, timeout=500)
    println("Best params is $(study.best_params) with value $(study.best_value)")
    return study
end

lbfgs_study()
```

    Best params is {'mem_len': 2, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 5.177946035544975e-05, 'Wolfe-Powell 1-c2': 0.00022065699240592798} with value 5529.393441724406

    [I 2024-03-08 20:47:28,906] A new study created in memory with name: L-BFGS
    [I 2024-03-08 20:48:00,952] Trial 0 finished with value: 6842.352584581565 and parameters: {'mem_len': 20, 'vector_transport_method': 1, 'retraction_method': 2, 'manifold': 1, 'manopt_stepsize': 1}. Best is trial 0 with value: 6842.352584581565.
    [I 2024-03-08 20:48:29,581] Trial 1 finished with value: 5770.31887029587 and parameters: {'mem_len': 23, 'vector_transport_method': 1, 'retraction_method': 2, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0006037261743456827, 'Wolfe-Powell 1-c2': 0.005132655740564769}. Best is trial 1 with value: 5770.31887029587.
    [I 2024-03-08 20:48:58,405] Trial 2 finished with value: 5739.761751248307 and parameters: {'mem_len': 28, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0006929848068517475, 'Wolfe-Powell 1-c2': 0.00010072848265341218}. Best is trial 2 with value: 5739.761751248307.
    [I 2024-03-08 20:49:25,419] Trial 3 finished with value: 5730.88644172447 and parameters: {'mem_len': 13, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0003147799818520074, 'Wolfe-Powell 1-c2': 0.0014392668301114121}. Best is trial 3 with value: 5730.88644172447.
    [I 2024-03-08 20:49:56,034] Trial 4 finished with value: 7719.590834581578 and parameters: {'mem_len': 30, 'vector_transport_method': 2, 'retraction_method': 2, 'manifold': 1, 'manopt_stepsize': 1}. Best is trial 3 with value: 5730.88644172447.
    [I 2024-03-08 20:50:12,154] Trial 5 pruned. 
    [I 2024-03-08 20:50:22,778] Trial 6 pruned. 
    [I 2024-03-08 20:50:49,949] Trial 7 finished with value: 5747.307298867354 and parameters: {'mem_len': 30, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0004374468447104617, 'Wolfe-Powell 1-c2': 0.0034326269868046764}. Best is trial 3 with value: 5730.88644172447.
    [I 2024-03-08 20:51:06,074] Trial 8 pruned. 
    [I 2024-03-08 20:51:23,029] Trial 9 pruned. 
    [I 2024-03-08 20:51:27,791] Trial 10 pruned. 
    [I 2024-03-08 20:51:54,957] Trial 11 finished with value: 5715.3223702958485 and parameters: {'mem_len': 11, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 1.214650115543274e-05, 'Wolfe-Powell 1-c2': 0.00010874873474069736}. Best is trial 11 with value: 5715.3223702958485.
    [I 2024-03-08 20:52:21,759] Trial 12 finished with value: 5674.56137029584 and parameters: {'mem_len': 10, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 1.1496159292180335e-05, 'Wolfe-Powell 1-c2': 0.00010048102268691798}. Best is trial 12 with value: 5674.56137029584.
    [I 2024-03-08 20:52:47,500] Trial 13 finished with value: 5609.9961560101265 and parameters: {'mem_len': 6, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 1.041783651693428e-05, 'Wolfe-Powell 1-c2': 0.00011453985064919054}. Best is trial 13 with value: 5609.9961560101265.
    [I 2024-03-08 20:52:51,944] Trial 14 pruned. 
    [I 2024-03-08 20:53:18,615] Trial 15 finished with value: 5672.700441724388 and parameters: {'mem_len': 8, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 1.2631008943053551e-05, 'Wolfe-Powell 1-c2': 0.00023471299637034984}. Best is trial 13 with value: 5609.9961560101265.
    [I 2024-03-08 20:53:22,373] Trial 16 pruned. 
    [I 2024-03-08 20:53:47,189] Trial 17 finished with value: 5529.393441724406 and parameters: {'mem_len': 2, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 5.177946035544975e-05, 'Wolfe-Powell 1-c2': 0.00022065699240592798}. Best is trial 17 with value: 5529.393441724406.
    [I 2024-03-08 20:53:51,629] Trial 18 pruned. 
    [I 2024-03-08 20:54:16,456] Trial 19 finished with value: 5529.489941724406 and parameters: {'mem_len': 2, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 4.7581369525514006e-05, 'Wolfe-Powell 1-c2': 0.00022584968387296984}. Best is trial 17 with value: 5529.393441724406.
    [I 2024-03-08 20:54:20,222] Trial 20 pruned. 
    [I 2024-03-08 20:54:23,985] Trial 21 pruned. 
    [I 2024-03-08 20:54:48,819] Trial 22 finished with value: 5529.415727438693 and parameters: {'mem_len': 2, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 8.613616174126538e-05, 'Wolfe-Powell 1-c2': 0.0004565420978727029}. Best is trial 17 with value: 5529.393441724406.
    [I 2024-03-08 20:54:52,592] Trial 23 pruned. 
    [I 2024-03-08 20:54:56,341] Trial 24 pruned. 
    [I 2024-03-08 20:55:00,090] Trial 25 pruned. 
    [I 2024-03-08 20:55:05,051] Trial 26 pruned. 
    [I 2024-03-08 20:55:08,813] Trial 27 pruned. 
    [I 2024-03-08 20:55:12,557] Trial 28 pruned. 
    [I 2024-03-08 20:55:18,145] Trial 29 pruned. 
    [I 2024-03-08 20:55:21,905] Trial 30 pruned. 
    [I 2024-03-08 20:55:25,665] Trial 31 pruned. 
    [I 2024-03-08 20:55:29,438] Trial 32 pruned. 
    [I 2024-03-08 20:55:54,460] Trial 33 finished with value: 5537.201441724406 and parameters: {'mem_len': 4, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.00020342007235269214, 'Wolfe-Powell 1-c2': 0.00015592126998615713}. Best is trial 17 with value: 5529.393441724406.

    Python: <optuna.study.study.Study object at 0x7f21bdfc0a10>

## Summary

We’ve shown how to automatically select the best hyperparameter values for your optimization problem.

## Literature

```@bibliography
Pages = ["HyperparameterOptimization.md"]
Canonical=false
```
