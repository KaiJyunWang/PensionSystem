# Run with Julia's environment containing HighVoronoi, QuasiMonteCarlo, CSV and Roots.
# Including this file defines helpers; running it directly solves the model and
# writes nodal values and fitted values along the pre-finance data path to table/.
using Roots, LogExpFunctions, CSV, DataFrames, Dates, Printf
include(joinpath(@__DIR__, "generator.jl"))

"""The economic model in stochastic_pension_run.jl, without its FD matrices."""
function voronoi_model(; b=0.00462, m=0.058, T=65, r=0.04, p=2.556,
        l=20.61, τ=0.125, y=5.496, α=-1.0, β=3.0, ρ=0.02, σ=0.08, B_max=10.0)
    m > 0 && ρ + m > 0 && β > 0 && σ >= 0 && B_max > 0 ||
        throw(ArgumentError("invalid model parameters"))
    n = find_zero(n -> (-expm1(-n*T))/n + exp(-n*T)/(n+m) - 1/b,
                  (-m + 1e-8, -1e-8))
    q(V) = V == 0 ? logistic(-α/β) : logistic(-(α + l - V)/β)
    μB(B,Q,V) = (r-n)*B + b*τ*y/n*(exp(-n*20)-exp(-n*T)) -
                (1-q(V))*b*exp(-n*T)*l - p*Q
    σB(B) = σ*B
    μQ(Q,V) = q(V)*b*exp(-n*T) - (m+n)*Q
    Q_max = b*exp(-n*T)/(m+n)
    return (; b,m,T,r,p,l,τ,y,α,β,ρ,σ,B_max,n,q,μB,σB,μQ,Q_max)
end

"""
Load observations strictly before `ext_finance_date` (default 2020-01-01).
As in stochastic_pension_run.jl, skip the first two source rows by default.
Filtering occurs before constructing the mesh; post-cutoff data never sets it.
"""
function load_voronoi_data(; path=joinpath(@__DIR__, "..", "data", "labor_insurance_fund_monthly.csv"),
        ext_finance_date=Date(2020,1,1), skiprows=2)
    df = CSV.read(path, DataFrame)
    0 <= skiprows < nrow(df) || throw(ArgumentError("invalid skiprows"))
    dates = [x isa Date ? x : Date(string(x), dateformat"yyyy-mm") for x in df.date]
    issorted(dates) || throw(ArgumentError("data dates must be sorted"))
    keep = [i for i in skiprows+1:nrow(df) if dates[i] < ext_finance_date]
    isempty(keep) && throw(ArgumentError("no observations before ext_finance_date"))
    sample = df[keep, :]
    sample.date = dates[keep]
    all(x -> !ismissing(x) && isfinite(x) && x > 0, sample.taiwan_registered_population) ||
        throw(ArgumentError("invalid population in pre-finance data"))
    sample.B = sample.fund_level_ntd ./ sample.taiwan_registered_population ./ 100_000
    sample.Q = sample.labor_insurance_old_age_pension_recipients ./ sample.taiwan_registered_population
    X_path = [[B,Q] for (B,Q) in zip(sample.B, sample.Q)]
    all(x -> all(v -> !ismissing(v) && isfinite(v) && v >= 0, x), X_path) ||
        throw(ArgumentError("invalid pre-finance B or Q observations"))
    return (; sample, X_path, ext_finance_date)
end

"""
Generate a uniform two-dimensional Sobol design over the model domain.
`X_path` is retained in the interface but does not affect node placement.
"""
pension_voronoi_grid(X_path; para, kwargs...) = data_voronoi_grid(X_path; para, kwargs...)

"""
Backward generator from the mass adjoint: A = F'. For the density generator
G = M⁻¹ F M, the equivalent weighted adjoint is A = M⁻¹ G' M, M=diag(volumes).
"""
construct_backward_generator(V; para, grid) =
    sparse(adjoint(construct_forward_generator(V; para, grid, representation=:mass)))

# Piecewise-linear interpolation on HighVoronoi's Delaunay dual. Outside the
# domain use flat upper-B/Q extrapolation and linear continuation below B=0.
function voronoi_value_interpolant(values, grid)
    length(values) == length(grid.nodes) || throw(DimensionMismatch("values must match nodes"))
    nodal_values = copy(values)
    function inside(B,Q)
        owners,weights = scattered_weights(grid.mesh,B/grid.B_max,Q/grid.Q_max)
        return sum(j == 0 ? 0.0 : w*nodal_values[j] for (j,w) in zip(owners,weights))
    end
    lower_reference = minimum(first.(grid.nodes))/2
    function V(B,Q)
        isfinite(B) && isfinite(Q) || throw(ArgumentError("nonfinite evaluation point"))
        q = clamp(Q,0,grid.Q_max)
        B == 0 && return 0.0
        B < 0 && return B/lower_reference*inside(lower_reference,q)
        return inside(min(B,grid.B_max),q)
    end
    return V
end

"""
Solve `(ρ+m)V - A(V)V = p` by damped fixed-point iteration on interior cells.
The lower Dirichlet condition is already in A, so every interior RHS is p.
Convergence requires both the update and the nonlinear residual to pass tol;
nonconvergence raises an error. Returns callable V, nodal values, and diagnostics.
"""
function solve_voronoi_pde(; para, grid, tol=1e-8, iterations=1000, damping=0.5,
        initial=nothing, verbose=true)
    isfinite(tol) && tol > 0 && iterations > 0 && 0 < damping <= 1 ||
        throw(ArgumentError("invalid solver controls"))
    discount = para.ρ + para.m
    discount > 0 || throw(ArgumentError("ρ+m must be positive"))
    N = length(grid.nodes)
    values = isnothing(initial) ? [para.p/discount * x[1]/grid.B_max for x in grid.nodes] : Float64.(vec(initial))
    length(values) == N || throw(DimensionMismatch("initial values must match the grid"))
    all(isfinite, values) || throw(ArgumentError("nonfinite initial values"))
    rhs = fill(para.p, N)
    A = construct_backward_generator(values; para, grid)
    residual = Inf
    for iteration in 1:iterations
        candidate = (discount*I - A) \ rhs
        updated = (1-damping).*values + damping.*candidate
        all(isfinite, updated) || error("nonfinite value iterate")
        update = norm(updated - values, Inf)
        values = updated
        A = construct_backward_generator(values; para, grid)
        residual = norm(discount*values - A*values - rhs, Inf)
        if verbose && (iteration == 1 || iteration % 25 == 0)
            @printf "Iteration %d: update %.3e, residual %.3e\n" iteration update residual
        end
        if update <= tol && residual <= tol
            V = voronoi_value_interpolant(values, grid)
            return (; V, values, A, F=sparse(adjoint(A)), grid, iterations=iteration, residual)
        end
    end
    error("Voronoi value iteration did not converge in $iterations iterations; residual=$residual")
end

"""
Solve using only pre-finance observations; optionally export two CSV tables.

```julia
include("stochastic_voronoi_run.jl")
result = run_voronoi_pension(; ext_finance_date=Date(2020,1,1))
V = result.solution.V
V(0.08, 0.01)
# Control mesh size with, for example: grid_options=(npoints=2048,)
# Use output_dir=nothing to solve without writing tables.
# Pass model parameters via para=voronoi_model(r=0.02, σ=0.1).
```
"""
function run_voronoi_pension(; para=voronoi_model(), ext_finance_date=Date(2020,1,1),
        data_options=(;), grid_options=(;), solver_options=(;),
        output_dir=joinpath(@__DIR__, "..", "table"))
    observations = load_voronoi_data(; ext_finance_date, data_options...)
    grid = pension_voronoi_grid(observations.X_path; para, grid_options...)
    solution = solve_voronoi_pde(; para, grid, solver_options...)
    if !isnothing(output_dir)
        mkpath(output_dir)
        CSV.write(joinpath(output_dir, "voronoi_value.csv"), DataFrame(
            B=first.(grid.nodes), Q=last.(grid.nodes), volume=grid.volumes, V=solution.values))
        fitted = copy(observations.sample)
        fitted.V = solution.V.(fitted.B, fitted.Q)
        fitted.q = para.q.(fitted.V)
        CSV.write(joinpath(output_dir, "voronoi_prefinance.csv"), fitted)
    end
    return (; para, observations, solution)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_voronoi_pension();
    @printf "Solved %d Voronoi cells in %d iterations; residual %.3e\n" length(result.solution.values) result.solution.iterations result.solution.residual
end

# plot functionals for visualization
#=
using Plots
Q_grids = range(0.0, result.para.Q_max, 201)
B_grids = range(0.0, result.para.B_max, 201)
Vs = [result.solution.V(B, Q) for B in B_grids, Q in Q_grids]
surface(Q_grids, B_grids, Vs, c=:viridis, alpha=0.5)
contourf(Q_grids, B_grids, Vs, c=:viridis)
begin 
    draw2D(result.solution.grid.geometry)
    plot!(aspect_ratio=:auto, xlims=(0.0,result.solution.grid.B_max),
        ylims=(0.0,result.solution.grid.Q_max),xlabel="B",ylabel="Q",size=(900,600))
end
=#