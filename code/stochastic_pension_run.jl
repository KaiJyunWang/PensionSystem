using CairoMakie, Roots, LogExpFunctions, LinearAlgebra
using DiffEqCallbacks, OrdinaryDiffEq, StochasticDiffEq
using Parameters, Printf, SteadyStateDiffEq
using DifferentialEquations, Interpolations
using Statistics, SparseArrays, ExponentialAction
using Optimization, OptimizationOptimJL, Random
using CSV, DataFrames, Dates, LaTeXStrings
import CairoMakie.Makie: date_to_number

set_theme!(
    fontsize = 18,
    Axis = (
        xgridvisible = false,
        ygridvisible = false,
        titlesize = 24,
        xlabelsize = 20,
        ylabelsize = 20,
        xticklabelsize = 16,
        yticklabelsize = 16,
    ),
    Axis3 = (
        xgridvisible = false,
        ygridvisible = false,
        zgridvisible = false,
        titlesize = 24,
        xlabelsize = 20,
        ylabelsize = 20,
        zlabelsize = 20,
        xticklabelsize = 16,
        yticklabelsize = 16,
        zticklabelsize = 16,
    ),
    Legend = (labelsize = 17,),
    Colorbar = (labelsize = 20, ticklabelsize = 16,),
)

include("parameter_tables.jl")
using .ParameterTables: write_parameter_table

function model(; b = 0.00462, m = 0.06, T = 61, r = 0.093, p = 2.449, Tw = 25, Tm = 60,
    l = 19.755, τ = 0.075, y = 5.268, α = -0.035, β = 1.0, ρ = 0.02, σ = 0.133,
    B_max = 10.0, grid_power = 2.0)
    # find population growth rate 
    n = find_zero(n -> (-expm1(-n * Tm)) / n + exp(-n * Tm) / (n + m) - 1 / b, (-m+ 1e-8, -1e-8))

    # auxilary functions
    g(s) = s < Tm ? b * exp(-n * s) : b * exp(-n * s - m * (s - Tm)) 
    labor_force = b * ((exp(-n*Tw) - exp(-n*Tm))/n + exp(m*Tm) * (exp(-(m+n)*Tm) - exp(-(m+n)*T)) / (m+n))
    q(V) = α != -Inf ? logistic(-(α + log(l) - log(max(V, 0))) / β) : 1.0
    μB(B, Q, V) = (r-n) * B + τ * y * labor_force - (1 - q(V)) * g(T) * l - p * Q
    σB(B) = σ * B
    μQ(Q, V) = q(V) * g(T) - (m+n) * Q

    Q_max = g(T) / (m + n) 

    # precomputed differentiation matrix
    nB = 200
    nQ = 80
    grid_power >= 1 || throw(ArgumentError("grid_power must be at least 1"))
    # The power map gives smaller cells near zero while retaining both endpoints.
    Bs = B_max .* (range(0.0, 1.0, length = nB) .^ grid_power)
    Qs = Q_max .* (range(0.0, 1.0, length = nQ) .^ grid_power)

    dB = diff(Bs)
    dQ = diff(Qs)

    IB = sparse(I, nB, nB)
    IQ = sparse(I, nQ, nQ)

    fDB = spdiagm(0 => vcat(-1 ./ dB, 0.0), 1 => 1 ./ dB)
    bDB = spdiagm(0 => vcat(0.0, 1 ./ dB), -1 => -1 ./ dB)
    fDQ = spdiagm(0 => vcat(-1 ./ dQ, 0.0), 1 => 1 ./ dQ)
    bDQ = spdiagm(0 => vcat(0.0, 1 ./ dQ), -1 => -1 ./ dQ)

    # Three-point second derivative on the nonuniform B grid.
    lower_B = 2.0 ./ (dB[1:end-1] .* (dB[1:end-1] .+ dB[2:end]))
    diag_B = -2.0 ./ (dB[1:end-1] .* dB[2:end])
    upper_B = 2.0 ./ (dB[2:end] .* (dB[1:end-1] .+ dB[2:end]))
    DBB = spdiagm(
        -1 => vcat(lower_B, 1 / dB[end]^2),
        0 => vcat(0.0, diag_B, -1 / dB[end]^2),
        1 => vcat(0.0, upper_B),
    )

    # adjust for boundary condition
    fDB[1, :] .= 0.0
    bDB[1, :] .= 0.0
    DBB[1, :] .= 0.0
    fDB[end, :] .= 0.0

    # kron for 2d grids 
    fDB = kron(IQ, fDB)
    bDB = kron(IQ, bDB)
    DBB = kron(IQ, DBB)
    fDQ = kron(fDQ, IB)
    bDQ = kron(bDQ, IB)

    return (; b, m, T, Tw, Tm, r, ρ, p, l, τ, y, n, α, β, σ, g, q, μB, σB, μQ, B_max, Q_max,
        grid_power,
        fDB, bDB, DBB, fDQ, bDQ, Bs, Qs)
end


# helper function to return discretized infinitesimal generator
function construct_generator(V; para)
    @unpack μB, σB, μQ, fDB, bDB, DBB, fDQ, bDQ, Bs, Qs = para

    MμB = spdiagm(0 => [μB(B, Q, V[i,j]) for (i, B) in enumerate(Bs), (j, Q) in enumerate(Qs)] |> vec)
    MμQ = spdiagm(0 => [μQ(Q, V[i,j]) for (i, B) in enumerate(Bs), (j, Q) in enumerate(Qs)] |> vec)
    MσB = spdiagm(0 => [0.5 * σB(B)^2 for (i, B) in enumerate(Bs), (j, Q) in enumerate(Qs)] |> vec)

    return max.(MμB, 0) * fDB + min.(MμB, 0) * bDB + max.(MμQ, 0) * fDQ + min.(MμQ, 0) * bDQ + MσB * DBB
end

# solve the PDV of the pension
function solve_pde(; tol = 1e-8, iterations = 1000, para, damp = 1.0)
    @unpack m, r, ρ, p, μB, σB, μQ, B_max, Q_max, Bs, Qs = para

    V0 = [p/(ρ+m) * B / B_max for B in Bs, Q in Qs]
    rhs = fill(p, size(V0))
    rhs[1, :] .= 0.0
    rhs = vec(rhs)
    
    iter = 1
    sup_norm = Inf 
    while iter ≤ iterations && sup_norm > tol 
        A = construct_generator(V0; para = para)
        V = (((ρ+m) * I - A) \ rhs) |> (x -> reshape(x, size(V0)))
        sup_norm = maximum(abs, V - V0)
        if iter % iterations == 0
            @printf "Iterations: %d \t Sup-norm: %.5g \n" iter sup_norm
        end
        iter += 1
        V0 = V * damp + V0 * (1 - damp)
    end

    return extrapolate(interpolate((Bs, Qs,), V0, Gridded(Linear())), Line())
end

# Solver for Kolmogorov Forward Equation. Take m0 to be a matrix to solve for multiple initial distribution 
function solve_kfe(Δ, m0; para, V)
    A = construct_generator(V; para = para)
    return expv(Δ, A', m0) |> (x -> reshape(x, length(para.Bs), length(para.Qs)))
end

cb = ContinuousCallback(
    (u, t, integrator) -> u[1],  
    nothing,
    (integrator) -> begin
        terminate!(integrator)
    end, 
    save_positions = (true, true)
)

#=
V = solve_pde(; para = para)

Q_grids = range(0.0, para.Q_max, 101)
B_grids = range(0.0, para.B_max, 101)
Vs = [V(B, Q) for B in B_grids, Q in Q_grids]
fig = Figure(size = (1200, 500))
ax3d = Axis3(fig[1, 1], xlabel = "Q", ylabel = "B", zlabel = "V", title = "V", azimuth = 50°, elevation = 30°)
surface!(ax3d, Q_grids, B_grids, permutedims(Vs), colormap = :viridis, transparency = true, alpha = 0.7)
ax2d = Axis(fig[1, 2], xlabel = "Q", ylabel = "B", title = "V")
contour_plot = contourf!(ax2d, Q_grids, B_grids, permutedims(Vs), colormap = :viridis, levels = 20)
Colorbar(fig[1, 3], contour_plot, label = "V")

# simulation 
# u = [B, Q]
function drift!(du, u, p, t)
    @unpack μB, μQ = p

    du[1] = μB(u[1], u[2], V(u[1], u[2]))
    du[2] = μQ(u[2], V(u[1], u[2]))
end

function diffusion!(du, u, p, t)
    @unpack σB = p
    du[1] = σB(u[1])
    du[2] = 0.0
end

cb = ContinuousCallback(
    (u, t, integrator) -> u[1],  
    nothing,
    (integrator) -> begin
        terminate!(integrator)
    end, 
    save_positions = (true, true)
)

sde_prob = SDEProblem(drift!, diffusion!, [0.084, 0.0], (0.0, 30.0), para)
sim = solve(sde_prob, SRIW1(), callback = cb, abstol = 1e-8, seed = 568)

# plot simulation

begin
    ts = range(0.0, sim.t[end], 300)
    sim_Vs = map(u -> V(u...), sim.(ts))
    fig = Figure(size = (600, 1000))
    axes = [Axis(fig[i, 1], title = title, xlabel = "t", ylabel = title) for (i, title) in enumerate(("B", "Q", "q", "V"))]
    lines!(axes[1], sim.t, getindex.(sim.u, 1))
    hlines!(axes[1], 0.0, color = :black, linestyle = :dash)
    lines!(axes[2], sim.t, getindex.(sim.u, 2))
    lines!(axes[3], ts, para.q.(sim_Vs))
    lines!(axes[4], ts, sim_Vs)
    xlims!.(axes[3:4], 0.0, sim.t[end])
    fig
end
=#

#---------------------------------------
# Estimation

# Least Square + Euler-Maruyama
# S is the decomposition such that W = S'S.
Δ = 1/12
function loss(θ; B_path = B_path, Q_path = Q_path, q_path = q_path, S = I, Δ = Δ)
    # para = model(; σ = exp(θ[1]), α = θ[2], β = θ[3])
    para = model(; σ = exp(θ[1]), α = θ[2], ρ = exp(θ[3]))
    V = solve_pde(; para = para)
    μB_targets = (diff(B_path) - Δ * para.μB.(B_path[1:end-1], Q_path[1:end-1], V.(B_path[1:end-1], Q_path[1:end-1])))./(sqrt(Δ) * B_path[1:end-1])
     

    σ_target = log(std(μB_targets)) - log(para.σ)
    q_mean = para.q.(V.(B_path[1:end-1], Q_path[1:end-1])) - q_path[1:end-1]

    ms = vcat(σ_target, q_mean)
    return mean(abs2, S * ms)
end

# Estimation with real data 
begin
    df = CSV.read("data/labor_insurance.csv", DataFrame)
    df[!, :date] = Date.(String.(df.date), dateformat"yyyy-mm")

    df[!, :B] = df.fund_level_ntd ./ df.taiwan_registered_population ./ 100_000
    df[!, :Q] = df.labor_insurance_old_age_pension_recipients ./ df.taiwan_registered_population
    df[!, :q] = df.new_monthly_pension_recipients ./ (df.new_monthly_pension_recipients + df.lumpsum_recipients)
    B_path = df.B
    Q_path = df.Q
    q_path = df.q
    # information shocks
    dates = [Date(2012, 9), Date(2020, 3)]
    event_labels = [
        "Reported Bankruptcy \n Time Shock",
        "Extra Funding",
    ]
end

# validate estimated new recipients 
begin
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "New Monthly Pension Recipients (thousand people)")
    estimated = .!ismissing.(df.estimated_new_monthly_pension_recipients)
    lines!(
        ax,
        df.date[estimated],
        Float64.(df.estimated_new_monthly_pension_recipients[estimated]) ./ 1_000;
        label = "Estimated",
        color = :brown,
    )
    reported = .!ismissing.(df.reported_new_monthly_pension_recipients)
    lines!(
        ax,
        df.date[reported],
        Float64.(df.reported_new_monthly_pension_recipients[reported]) ./ 1_000;
        label = "Data",
        color = :black,
    )
    axislegend(ax)
    display(fig)
    save("./figure/validate_reconstruction.png", fig, px_per_unit = 2)
end

# compare the absolute amount of withdrawals
begin
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "New Monthly Pension Recipients (thousand people)")
    estimated = .!ismissing.(df.estimated_new_monthly_pension_recipients)
    lines!(
        ax,
        df.date[estimated],
        Float64.(df.estimated_new_monthly_pension_recipients[estimated]) ./ 1_000;
        label = "Estimated",
        color = :brown,
    )
    reported = .!ismissing.(df.reported_new_monthly_pension_recipients)
    lines!(
        ax,
        df.date[reported],
        Float64.(df.reported_new_monthly_pension_recipients[reported]) ./ 1_000;
        label = "Data",
        color = :black,
    )
    axislegend(ax)
    display(fig)
    save("./figure/validate_reconstruction.png", fig, px_per_unit = 2)
end

begin
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "Number of Recipients (thousand people)")
    lines!(ax, df.date, df.new_monthly_pension_recipients / 1000, label = "Monthly")
    lines!(ax, df.date, df.lumpsum_recipients / 1000, label = "Lumpsum")
    event_xs = date_to_number.(DateTime, dates)
    sample_end_x = date_to_number(DateTime, maximum(df.date))
    vlines!(ax, range(event_xs[2], sample_end_x, length = 300), color = (:teal, 0.15), linewidth = 2)
    vlines!(ax, event_xs, color = :black, linestyle = :dash)
    label_y = 45
    ylims!(ax, 0.0, 50.0)
    for (date, label) in zip(dates, event_labels)
        textlabel!(
            ax,
            date,
            label_y;
            text = label,
            text_align = (:left, :bottom),
            offset = (8, 0),
            fontsize = 18,
            background_color = (:white, 0.85),
            strokewidth = 0,
            padding = 3,
        )
    end
    axislegend(ax)
    display(fig)
    save("./figure/recipients.png", fig, px_per_unit = 2)
end

# plot the recipients fractions
begin
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "Fraction of Monthly")
    lines!(ax, df.date, df.q, color = :black)
    event_xs = date_to_number.(DateTime, dates)
    sample_end_x = date_to_number(DateTime, maximum(df.date))
    vlines!(ax, range(event_xs[2], sample_end_x, length = 300), color = (:teal, 0.15), linewidth = 2)
    vlines!(ax, event_xs, color = :black, linestyle = :dash)
    label_y = 0.1
    ylims!(ax, 0.0, 1.0)
    for (date, label) in zip(dates, event_labels)
        textlabel!(
            ax,
            date,
            label_y;
            text = label,
            text_align = (:left, :bottom),
            offset = (8, 0),
            fontsize = 18,
            background_color = (:white, 0.85),
            strokewidth = 0,
            padding = 3,
        )
    end
    display(fig)
    save("./figure/recipient_fraction.png", fig, px_per_unit = 2)
end


begin
   fig = Figure(size = (800, 600))
   ax1 = Axis(fig[1, 1], title = "Fund Level (100,000 NTD) / Population")
   ax2 = Axis(fig[2, 1], title = "Recipients / Population")
   lines!(ax1, df.date, df.B, color = :black)
   lines!(ax2, df.date, df.Q, color = :black)
   event_xs = date_to_number.(DateTime, dates)
   sample_end_x = date_to_number(DateTime, maximum(df.date))
   funding_period_xs = range(event_xs[2], sample_end_x, length = 300)
   vlines!(ax1, funding_period_xs, color = (:teal, 0.15), linewidth = 2)
   vlines!(ax2, funding_period_xs, color = (:teal, 0.15), linewidth = 2)
   vlines!(ax1, event_xs, color = :black, linestyle = :dash)
   vlines!(ax2, event_xs, color = :black, linestyle = :dash)
   for (ax, values) in ((ax1, df.B), (ax2, df.Q))
       value_min, value_max = extrema(values)
       value_range = value_max - value_min
       label_ys = (value_max - 0.05value_range, value_min + 0.05value_range)
       vertical_alignments = (:top, :bottom)
       for (date, label, label_y, valign) in zip(dates, event_labels, label_ys, vertical_alignments)
           textlabel!(
               ax,
               date,
               label_y;
               text = label,
               text_align = (:left, valign),
               offset = (8, 0),
               fontsize = 18,
               background_color = (:white, 0.85),
               strokewidth = 0,
               padding = 3,
           )
       end
   end
   display(fig)
   save("./figure/state_data.png", fig, px_per_unit = 2)
end

# plot population density
begin
    ages = 0.0:1.0:100.0
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Age", ylabel = "Population density")
    lines!(ax, ages, model().g.(ages), color = :black)
    display(fig)
    save("./figure/pop_density.png", fig, px_per_unit = 2)
end

# initial guesses
# estimate r
r = mean(skipmissing(df.fund_annualized_return_pct))/100
σ0 = 0.12
α0 = 0.0
ρ0 = 0.02

θ0 = [log(σ0), α0, log(ρ0)]
# θ0 = [log(σ0), α0]

optf = OptimizationFunction((θ, p) -> loss(θ), AutoFiniteDiff())
prob = OptimizationProblem(optf, θ0)
sol = solve(prob, NelderMead(); show_trace = true)
res = sol.u |> (x -> [exp(x[1]), x[2], exp(x[3])])

para_est = model(; σ = res[1], α = res[2], ρ = res[3])

# output parameter table
write_parameter_table(para_est)
write_parameter_table(para_est; output_path = "table/parameters_A.tex", panel = :A)
write_parameter_table(para_est; output_path = "table/parameters_B.tex", panel = :B)


# simulate the estimated model
V_est = solve_pde(para = para_est, iterations = 1000, damp = 0.5)

Q_grids = range(0.0, para_est.Q_max, 301)
B_grids = range(0.0, para_est.B_max, 301)
V_ests = [V_est(B, Q) for B in B_grids, Q in Q_grids]
begin
    make_value_figure() = begin
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Q", ylabel = "B", title = "V")
    contour_plot = contourf!(ax, Q_grids, B_grids, permutedims(V_ests), colormap = :viridis, levels = 20)
    Colorbar(fig[1, 2], contour_plot)
        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1], xlabel = "Q", ylabel = "B", title = "V")
        contour_plot = contourf!(ax, Q_grids, B_grids, permutedims(V_ests), colormap = :viridis, levels = 20)
        Colorbar(fig[1, 2], contour_plot)
        return fig, ax
    end

    fig, ax = make_value_figure()
    display(fig)
    save("./figure/estimated_V.png", fig, px_per_unit = 2)

    q_mid = (first(Q_grids) + last(Q_grids)) / 2
    b_mid = (first(B_grids) + last(B_grids)) / 2
    q_span = last(Q_grids) - first(Q_grids)
    b_span = last(B_grids) - first(B_grids)

    fig_fund, ax_fund = make_value_figure()
    arrows2d!(
        ax_fund,
        [Point2f(q_mid, b_mid)],
        [Vec2f(0, 0.3b_span)];
        align = :center,
        color = :orangered,
        shaftwidth = 10,
        tipwidth = 30,
        tiplength = 24,
    )
    textlabel!(
        ax_fund,
        q_mid + 0.04q_span,
        b_mid;
        text = "Higher Fund Level",
        text_align = (:left, :center),
        fontsize = 20,
        background_color = (:white, 0.9),
        strokewidth = 0,
        padding = 4,
    )
    display(fig_fund)
    save("./figure/estimated_V_higher_fund.png", fig_fund, px_per_unit = 2)

    fig_recipients, ax_recipients = make_value_figure()
    arrows2d!(
        ax_recipients,
        [Point2f(q_mid, b_mid)],
        [Vec2f(0.3q_span, 0)];
        align = :center,
        color = :orangered,
        shaftwidth = 10,
        tipwidth = 30,
        tiplength = 24,
    )
    textlabel!(
        ax_recipients,
        q_mid,
        b_mid + 0.06b_span;
        text = "More Monthly Recipients",
        text_align = (:center, :bottom),
        fontsize = 20,
        background_color = (:white, 0.9),
        strokewidth = 0,
        padding = 4,
    )
    display(fig_recipients)
    save("./figure/estimated_V_more_monthly_recipients.png", fig_recipients, px_per_unit = 2)
end

# validation for q 
begin
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "q")
    model_q = para_est.q.(V_est.(df.B, df.Q))
    lines!(ax, df.date, df.q, label = "Data", color = :black)
    lines!(ax, df.date, model_q, label = "Model", color = :brown)
    event_xs = date_to_number.(DateTime, dates)
    sample_end_x = date_to_number(DateTime, maximum(df.date))
    vlines!(ax, range(event_xs[2], sample_end_x, length = 300), color = (:teal, 0.15), linewidth = 2)
    vlines!(ax, event_xs, color = :black, linestyle = :dash)
    q_min = min(minimum(df.q), minimum(model_q))
    q_max = max(maximum(df.q), maximum(model_q))
    q_range = q_max - q_min
    label_ys = (q_max - 0.05q_range, q_min + 0.05q_range)
    vertical_alignments = (:top, :bottom)
    for (date, label, label_y, valign) in zip(dates, event_labels, label_ys, vertical_alignments)
        textlabel!(
            ax,
            date,
            label_y;
            text = label,
            text_align = (:left, valign),
            offset = (8, 0),
            fontsize = 18,
            background_color = (:white, 0.85),
            strokewidth = 0,
            padding = 3,
        )
    end
    axislegend(ax; position = :rb)
    display(fig)
    save("./figure/validation_q.png", fig, px_per_unit = 2)
end


# simulation 
# defaul initial condition to 2026/06
function simulate(para; 
    x0 = [df.B[end], df.Q[end]], 
    tspan = (0.0, 10.0), n_sim = 2000, seed = 2026, 
    output_func = (sol, ctx) -> (sol, false), damp = 0.5)

    V_est = solve_pde(para = para, damp = damp)
    function drift_est!(du, u, p, t)
        @unpack μB, μQ = p

        du[1] = μB(u[1], u[2], V_est(u[1], u[2]))
        du[2] = μQ(u[2], V_est(u[1], u[2]))
    end

    function diffusion_est!(du, u, p, t)
        @unpack σB = p
        du[1] = σB(u[1])
        du[2] = 0.0
    end
    sde_prob = SDEProblem(drift_est!, diffusion_est!, x0, tspan, para)

    Random.seed!(seed)
    seeds = rand(UInt64, n_sim)

    ensemble_prob = EnsembleProblem(
        sde_prob, 
        output_func = output_func,
        prob_func = (prob, ctx) -> remake(prob; seed = seeds[ctx.sim_id])
    )
    return solve(
            ensemble_prob,
            SRIW1(),
            EnsembleThreads();
            trajectories = n_sim, 
            callback = cb, 
            abstol = 1e-8
        )
end

# simulate bankruptcy time
sim_benchmark = simulate(para_est, output_func = (sol, ctx) -> (sol.t[end], false))
sim_all_lumpsum = simulate(model(σ = res[1], α = Inf, ρ = res[3]), output_func = (sol, ctx) -> (sol.t[end], false))
sim_all_monthly = simulate(model(σ = res[1], α = -Inf, ρ = res[3]), output_func = (sol, ctx) -> (sol.t[end], false))

begin
    times = 0.0:1/12:10.0
    bankrupt_prob = [mean(sim_benchmark.u .< t) for t in times]
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Year", ylabel = "Bankruptcy probability", title = "Bankruptcy Probability from Jun., 2026")
    lines!(ax, times, bankrupt_prob, label = "Benchmark")
    bankrupt_prob = [mean(sim_all_lumpsum.u .< t) for t in times]
    lines!(ax, times, bankrupt_prob, label = "All Lumpsum")
    bankrupt_prob = [mean(sim_all_monthly.u .< t) for t in times]
    lines!(ax, times, bankrupt_prob, label = "All Monthly")
    axislegend(ax; position = :rb)
    display(fig)
    save("./figure/bankruptcy_prob.png", fig, px_per_unit = 2)
end

# simulate bankruptcy times for different tax level 
begin
    times = 0.0:1/12:50.0
    τs = 0.075:0.04:0.235
    nτ = length(τs)
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1], title = "Bankruptcy Probability from Jun., 2026", xlabel = "Year", ylabel = "Bankruptcy probability")
    for (i, τ) in enumerate(τs)
        para = model(σ = res[1], α = res[2], ρ = res[3], τ = τ)
        sim = simulate(para, output_func = (sol, ctx) -> (sol.t[end], false), tspan=(0.0, times[end]))
        bankrupt_prob = [mean(sim.u .< t) for t in times]
        lines!(ax, times, bankrupt_prob, label = "τ = $τ", color = RGBf(i/nτ, 0.2, 1-i/nτ))
    end
    Legend(fig[1, 2], ax)
    display(fig)
    save("./figure/bankruptcy_prob_tax.png", fig, px_per_unit = 2)
end

# simulate one path for mechanism explain 
begin
    x0 = [0.1, 0.0]
    τs = [0.075, 0.13]
    times = 0.0:0.1:50.0
    Q_ubs = [0.02, 0.3]
    B_ubs = [0.2, 5.5]
    mechanism_data = []

    # Compute each scenario once. The stored results are reused for all variants.
    for (i, τ) in enumerate(τs)
        para = model(σ = res[1], α = res[2], ρ = res[3], τ = τ)
        V = solve_pde(para = para)
        Q_grids = range(0.0, Q_ubs[i], 301)
        B_grids = range(0.0, B_ubs[i], 301)
        Vs = [V(B, Q) for B in B_grids, Q in Q_grids]
        sim = simulate(para; x0 = x0, n_sim = 1, tspan = (times[1], times[end]), seed = 302)
        trajectory = only(sim.u)
        path_times = times[times .<= trajectory.t[end]]
        path = hcat(trajectory.(path_times)...)
        peak_id = argmax(path[1, :])
        post_peak = peak_id:size(path, 2)
        drawdown_candidates = post_peak[path[1, post_peak] .<= 0.7path[1, peak_id]]
        drawdown_id = isempty(drawdown_candidates) ? peak_id : first(drawdown_candidates)
        stage_ids = [1, drawdown_id, size(path, 2)]
        stage_labels = ["Initial state", "Drawdown", "Bankruptcy"]
        stage_colors = [:white, :orange, :red]
        stage_valigns = [:bottom, :bottom, :bottom]
        stage_offsets = [(8, 8), (8, 8), (8, 8)]
        if peak_id > 1 && peak_id < size(path, 2)
            insert!(stage_ids, 2, peak_id)
            insert!(stage_labels, 2, "Peak fund level")
            insert!(stage_colors, 2, :gold)
            insert!(stage_valigns, 2, :top)
            insert!(stage_offsets, 2, (8, -8))
        end
        if i == 2
            early_id = argmin(abs.(path_times .- 5.0))
            insert!(stage_ids, 2, early_id)
            insert!(stage_labels, 2, "Early Recipients\n Choosing Monthly")
            insert!(stage_colors, 2, :deepskyblue)
            insert!(stage_valigns, 2, :bottom)
            insert!(stage_offsets, 2, (8, 8))
        end

        push!(mechanism_data, (; τ, Q_grids, B_grids, Vs, path, stage_ids,
            stage_labels, stage_colors, stage_valigns, stage_offsets))
    end

    function make_mechanism_figure(data, n_stages)
        fig = Figure(size = (900, 650))
        ax = Axis(fig[1, 1], xlabel = "Q", ylabel = "B",
            title = "Pension Tax Rate = $(data.τ)")
        contour_plot = contourf!(ax, data.Q_grids, data.B_grids,
            permutedims(data.Vs), colormap = :viridis, levels = 20)
        Colorbar(fig[1, 2], contour_plot, label = "V")

        path = data.path
        lines!(ax, path[2, :], path[1, :], color = (:white, 0.8), linewidth = 8)
        lines!(ax, path[2, :], path[1, :], color = :brown, linewidth = 5)

        arrow_ids = unique(round.(Int,
            range(2, size(path, 2) - 1, length = min(4, size(path, 2) - 2))))
        arrow_dx = path[2, arrow_ids .+ 1] .- path[2, arrow_ids .- 1]
        arrow_dy = path[1, arrow_ids .+ 1] .- path[1, arrow_ids .- 1]
        scatter!(ax, path[2, arrow_ids], path[1, arrow_ids]; marker = :utriangle,
            rotation = atan.(arrow_dy, arrow_dx) .- pi / 2, markersize = 15,
            color = :white, strokecolor = :brown, strokewidth = 2)

        if n_stages > 0
            ids = data.stage_ids[1:n_stages]
            scatter!(ax, path[2, ids], path[1, ids],
                color = data.stage_colors[1:n_stages], strokecolor = :black,
                strokewidth = 2, markersize = 14)
            for j in 1:n_stages
                id = data.stage_ids[j]
                textlabel!(ax, path[2, id], path[1, id];
                    text = data.stage_labels[j],
                    text_align = (:left, data.stage_valigns[j]),
                    offset = data.stage_offsets[j], fontsize = 17,
                    text_color = :black, background_color = (:white, 0.85),
                    strokewidth = 0, padding = 3)
            end
        end

        limits!(ax, first(data.Q_grids), last(data.Q_grids),
            first(data.B_grids), last(data.B_grids))
        return fig
    end

    for data in mechanism_data
        tax_slug = replace(string(data.τ), "." => "")
        base_fig = make_mechanism_figure(data, 0)
        save("./figure/mechanism_tax_$(tax_slug)_stage_0_trajectory.png",
            base_fig, px_per_unit = 2)

        for n_stages in eachindex(data.stage_ids)
            stage_slug = replace(lowercase(data.stage_labels[n_stages]),
                r"[^a-z]+" => "_") |> x -> strip(x, '_')
            stage_fig = make_mechanism_figure(data, n_stages)
            save("./figure/mechanism_tax_$(tax_slug)_stage_$(n_stages)_$(stage_slug).png",
                stage_fig, px_per_unit = 2)
        end

        final_fig = make_mechanism_figure(data, length(data.stage_ids))
        display(final_fig)
        save("./figure/mechanism_tax_$(tax_slug).png", final_fig, px_per_unit = 2)
    end
end

# simulation for cutting benefit
begin
    times = 0.0:1/12:50.0
    cuts = 0.0:0.1:0.6
    n = length(cuts)
    p_benchmark = model().p
    l_benchmark = model().l
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1], title = "Bankruptcy Probability from Jun., 2026", xlabel = "Year", ylabel = "Bankruptcy probability")
    for (i, cut) in enumerate(cuts)
        para = model(σ = res[1], α = res[2], ρ = res[3], p = (1-cut) * p_benchmark, l = (1-cut) * l_benchmark)
        sim = simulate(para, output_func = (sol, ctx) -> (sol.t[end], false), tspan=(0.0, times[end]))
        bankrupt_prob = [mean(sim.u .< t) for t in times]
        lines!(ax, times, bankrupt_prob, label = "-$(100*cut)%", color = RGBf(i/n, 0.2, 1-i/n))
    end
    Legend(fig[1, 2], ax)
    display(fig)
    save("./figure/bankruptcy_prob_cuts.png", fig, px_per_unit = 2)
end
