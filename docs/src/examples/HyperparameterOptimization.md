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

# TTsuggest_ structs collect data from a calibrating optimization run
# that is handled by compute_pruning_losses function

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

function compute_pruning_losses(
    od,
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
```

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
    selected_manifold = od.manifold_constructors[pyconvert(Int, study.best_params["manifold"])][1]
    selected_retraction_method = od.retrs[pyconvert(Int, study.best_params["retraction_method"])]
    selected_vector_transport = od.vts[pyconvert(Int, study.best_params["vector_transport_method"])]
    println("Selected manifold: $(selected_manifold)")
    println("Selected retraction method: $(selected_retraction_method)")
    println("Selected vector transport method: $(selected_vector_transport)")
    return study
end

lbfgs_study()
```

    Best params is {'mem_len': 3, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0006125542888545935, 'Wolfe-Powell 1-c2': 0.0010744467792321093} with value 5510.963227438757
    Selected manifold: Sphere
    Selected retraction method: ExponentialRetraction()
    Selected vector transport method: ProjectionTransport()

    [I 2024-03-16 18:04:17,965] A new study created in memory with name: L-BFGS
    [I 2024-03-16 18:04:45,996] Trial 0 finished with value: 5639.789870295856 and parameters: {'mem_len': 26, 'vector_transport_method': 1, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.00027288064367948073, 'Wolfe-Powell 1-c2': 0.00026503788892114045}. Best is trial 0 with value: 5639.789870295856.
    [I 2024-03-16 18:05:11,860] Trial 1 finished with value: 5635.936370295855 and parameters: {'mem_len': 11, 'vector_transport_method': 1, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.002743250060163298, 'Wolfe-Powell 1-c2': 0.00037986521186922096}. Best is trial 1 with value: 5635.936370295855.
    [I 2024-03-16 18:05:39,386] Trial 2 finished with value: 5673.101441724422 and parameters: {'mem_len': 26, 'vector_transport_method': 2, 'retraction_method': 2, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.00043339485784312605, 'Wolfe-Powell 1-c2': 0.0027302649933974173}. Best is trial 1 with value: 5635.936370295855.
    [I 2024-03-16 18:06:10,279] Trial 3 finished with value: 7410.818084581564 and parameters: {'mem_len': 26, 'vector_transport_method': 1, 'retraction_method': 2, 'manifold': 1, 'manopt_stepsize': 1}. Best is trial 1 with value: 5635.936370295855.
    [I 2024-03-16 18:06:37,995] Trial 4 finished with value: 5756.566226449636 and parameters: {'mem_len': 25, 'vector_transport_method': 1, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 1}. Best is trial 1 with value: 5635.936370295855.
    [I 2024-03-16 18:06:42,755] Trial 5 pruned. 
    [I 2024-03-16 18:06:58,577] Trial 6 pruned. 
    [I 2024-03-16 18:07:15,366] Trial 7 pruned. 
    [I 2024-03-16 18:07:40,605] Trial 8 finished with value: 5581.7437274386975 and parameters: {'mem_len': 7, 'vector_transport_method': 1, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0010567355712112379, 'Wolfe-Powell 1-c2': 0.003948002490203636}. Best is trial 8 with value: 5581.7437274386975.
    [I 2024-03-16 18:07:46,021] Trial 9 pruned. 
    [I 2024-03-16 18:08:11,512] Trial 10 finished with value: 5510.963227438757 and parameters: {'mem_len': 3, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0006125542888545935, 'Wolfe-Powell 1-c2': 0.0010744467792321093}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:08:35,914] Trial 11 finished with value: 5521.388656010121 and parameters: {'mem_len': 2, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0006738829952322474, 'Wolfe-Powell 1-c2': 0.0010639659137420014}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:09:00,317] Trial 12 finished with value: 5521.36958458155 and parameters: {'mem_len': 2, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.00010975606104676191, 'Wolfe-Powell 1-c2': 0.0007663843095951679}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:09:24,680] Trial 13 finished with value: 5520.7020845815505 and parameters: {'mem_len': 2, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 6.743450835567536e-05, 'Wolfe-Powell 1-c2': 0.0008779759729737719}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:09:50,268] Trial 14 pruned. 
    [I 2024-03-16 18:10:15,494] Trial 15 finished with value: 5595.119584581556 and parameters: {'mem_len': 6, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 8.147444451747575e-05, 'Wolfe-Powell 1-c2': 0.00012268601197923553}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:10:25,264] Trial 16 pruned. 
    [I 2024-03-16 18:10:50,209] Trial 17 finished with value: 5572.474513153012 and parameters: {'mem_len': 5, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.0015998473664092935, 'Wolfe-Powell 1-c2': 0.0005109172020536229}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:10:54,772] Trial 18 pruned. 
    [I 2024-03-16 18:11:04,534] Trial 19 pruned. 
    [I 2024-03-16 18:11:28,873] Trial 20 finished with value: 5512.3824417244705 and parameters: {'mem_len': 3, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 1.1581668103921961e-05, 'Wolfe-Powell 1-c2': 0.0002691056199427656}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:11:53,327] Trial 21 finished with value: 5529.088227438692 and parameters: {'mem_len': 4, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 1.3645031886009879e-05, 'Wolfe-Powell 1-c2': 0.0001863385753491203}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:12:17,911] Trial 22 finished with value: 5522.041370295835 and parameters: {'mem_len': 2, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 1.0030173525937465e-05, 'Wolfe-Powell 1-c2': 0.000543991948003312}. Best is trial 10 with value: 5510.963227438757.
    [I 2024-03-16 18:12:27,645] Trial 23 pruned. 
    [I 2024-03-16 18:12:52,163] Trial 24 finished with value: 5528.840941724406 and parameters: {'mem_len': 4, 'vector_transport_method': 2, 'retraction_method': 1, 'manifold': 1, 'manopt_stepsize': 2, 'Wolfe-Powell c1': 0.000245400433292576, 'Wolfe-Powell 1-c2': 0.000133639324295565}. Best is trial 10 with value: 5510.963227438757.

    Python: <optuna.study.study.Study object at 0x70dd985d9b50>

## Summary

We’ve shown how to automatically select the best hyperparameter values for your optimization problem.

## Literature

```@bibliography
Pages = ["HyperparameterOptimization.md"]
Canonical=false
```
