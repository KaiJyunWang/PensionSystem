using Plots, Roots, LogExpFunctions, LinearAlgebra
using DiffEqCallbacks, OrdinaryDiffEq, StochasticDiffEq
using Parameters, Printf, SteadyStateDiffEq
using DifferentialEquations, Interpolations
using Statistics, SparseArrays, ExponentialAction
using Optimization, OptimizationOptimJL, Random
using CSV, DataFrames, Dates

include("parameter_tables.jl")
using .ParameterTables: write_parameter_table

function model(; b = 0.00462, m = 0.06, T = 61, r = 0.057, p = 2.449, Tw = 25, Tm = 60,
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
surface(Q_grids, B_grids, Vs, xlabel = "Q", ylabel = "B", title = "V", camera = (50,30), alpha = 0.7, c=:viridis)
contourf(Q_grids, B_grids, Vs, xlabel = "Q", ylabel = "B", title = "V", c=:viridis, levels = 20, lw = 0)

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
    p1 = plot(sim, idxs = 1, title = "B", xlabel = "t", label = "")
    plot!(p1, sim.t, zeros(length(sim.t)), label = "", c=:black, ls=:dash)
    p2 = plot(sim, idxs = 2, title = "Q", xlabel = "t", label = "")
    p3 = plot(ts, para.q.(sim_Vs), title = "q", xlabel = "t", label = "", xlims = (0.0, sim.t[end]))
    p4 = plot(ts, sim_Vs, title = "V", xlabel = "t", label = "", xlims = (0.0, sim.t[end]))
    plt = plot(p1, p2, p3, p4, layout = (4, 1), size = (600, 1000))
end
=#

#---------------------------------------
# Estimation

# Least Square + Euler-Maruyama
# S is the decomposition such that W = S'S.
Δ = 1/12
function loss(θ; B_path = B_path, Q_path = Q_path, q_path = q_path, S = I, Δ = Δ)
    # para = model(; σ = exp(θ[1]), α = θ[2], β = θ[3])
    para = model(; σ = exp(θ[1]), α = θ[2], ρ = -0.06+1e-4+exp(θ[3]))
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

    ext_finance_date_id = findall(x -> x == Date(2020), df.date)|> only

    df[!, :B] = df.fund_level_ntd ./ df.taiwan_registered_population ./ 100_000
    df[!, :Q] = df.labor_insurance_old_age_pension_recipients ./ df.taiwan_registered_population
    df[!, :q] = df.new_monthly_pension_recipients ./ (df.new_monthly_pension_recipients + df.lumpsum_recipients)
    B_path = df.B[3:ext_finance_date_id-1]
    Q_path = df.Q[3:ext_finance_date_id-1]
    q_path = df.q[3:ext_finance_date_id-1] .|> Float64
end

# validate estimated new recipients 
begin
    plt = plot(df.date, df.estimated_new_monthly_pension_recipients, label = "Estimated")
    plot!(plt, df.date, df.reported_new_monthly_pension_recipients, label = "Data")
    display(plt)
    savefig(plt, "./figure/validate_reconstruction.png")
end

# plot the recipients 
begin
    p1 = plot(df.date[3:end], df.q[3:end], label="", c=:black, title="Proportion of Monthly")
    p2 = plot(df.date[3:end], df.new_monthly_pension_recipients[3:end], label="Monthly", leg=:topright)
    plot!(p2, df.date[3:end], df.lumpsum_recipients[3:end], label="Lumpsum", title="Number of New Recipients")
    vline!.([p1, p2], Ref([Date(2020)]), label="",  c=:black, ls=:dash)
    plt = plot(p1, p2, layout=(2,1), size=(800, 600))
    display(plt)
    savefig(plt, "./figure/recipients.png")
end


begin
   p1 = plot(df.date[3:end], df.B[3:end], label = "", title="Fund Level per Capita (100,000 NTD)", c=:black)
   p2 = plot(df.date[3:end], df.Q[3:end], label = "", title="Recipients/Population", c=:black)
   #p3 = plot(df.date[3:end-1], df.q[3:end-1], title="q", label="", c=:black)
   vline!.([p1, p2], Ref([Date(2020)]), label="",  c=:black, ls=:dash)
   plt = plot(p1, p2, layout = (2, 1), size = (800, 600))
   display(plt)
   savefig(plt, "./figure/state_data.png")
end

# plot population density
begin
    plt = plot(0.0:1.0:100.0, model().g, label="", xlabel="age", c=:black)
    display(plt)
    savefig(plt, "./figure/pop_density.png")
end

# initial guesses
# estimate r
r = mean(skipmissing(df.fund_annualized_return_pct[1:ext_finance_date_id-1]/100))
σ0 = 0.12
α0 = 0.0
ρ0 = 0.02

θ0 = [log(σ0), α0, log(ρ0+0.06)]
# θ0 = [log(σ0), α0]

optf = OptimizationFunction((θ, p) -> loss(θ), AutoFiniteDiff())
prob = OptimizationProblem(optf, θ0)
sol = solve(prob, NelderMead(); show_trace = true)
res = sol.u |> (x -> [exp(x[1]), x[2], -0.06+1e-4+exp(x[3])])

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
    p1 = surface(Q_grids, B_grids, V_ests, xlabel = "Q", ylabel = "B", title = "V", camera = (50,30), alpha = 0.7, c=:viridis)
    p2 = contourf(Q_grids, B_grids, V_ests, xlabel = "Q", ylabel = "B", title = "V", c=:viridis, levels = 20, lw = 0)
    plt = plot(p1, p2, layout=(1, 2), size=(800, 400))
    display(plt)
    savefig(plt, "./figure/estimated_V.png")
end

# validation for q 
begin
    plt = plot(df.date[3:end], df.q[3:end], title="q", label="Data", c=:black)
    vline!(plt, [Date(2020)], label="",  c=:black, ls=:dash)    
    plot!(plt, df.date[3:end], para_est.q.(V_est.(df.B, df.Q))[3:end], label="Model", c=:brown)
    display(plt)
    savefig(plt, "./figure/validation_q.png")
end


# simulation 
# defaul initial condition to 2020/01
function simulate(para; 
    x0 = [df.B[ext_finance_date_id], df.Q[ext_finance_date_id]], 
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
    plt = plot(times, bankrupt_prob, xlabel="year", label="Benchmark", title="Bankruptcy Probability from Jan., 2020")  
    bankrupt_prob = [mean(sim_all_lumpsum.u .< t) for t in times]
    plot!(plt, times, bankrupt_prob, xlabel="year", label="All Lumpsum", title="Bankruptcy Probability from Jan., 2020")
    bankrupt_prob = [mean(sim_all_monthly.u .< t) for t in times]
    plot!(plt, times, bankrupt_prob, xlabel="year", label="All Monthly", title="Bankruptcy Probability from Jan., 2020")
    display(plt)  
    savefig(plt, "./figure/bankruptcy_prob.png")
end

# simulate bankruptcy times for different tax level 
begin
    times = 0.0:1/12:50.0
    τs = 0.075:0.04:0.315
    nτ = length(τs)
    plt = plot(title = "Bankruptcy Probability from Jan., 2020", xlabel = "year", leg=:outerright)
    for (i, τ) in enumerate(τs)
        para = model(σ = res[1], α = res[2], ρ = res[3], τ = τ)
        sim = simulate(para, output_func = (sol, ctx) -> (sol.t[end], false), tspan=(0.0, times[end]))
        bankrupt_prob = [mean(sim.u .< t) for t in times]
        plot!(plt, times, bankrupt_prob, label="τ = $τ", color = RGB(i/nτ, 0.2, 1-i/nτ))
    end
    display(plt)
    savefig(plt, "./figure/bankruptcy_prob_tax.png")
end

# plot V under different tax rate 
begin
    τs = 0.075:0.04:0.315
    plts = []
    for (i, τ) in enumerate(τs)
        para = model(σ = res[1], α = res[2], ρ = res[3], τ = τ)
        V = solve_pde(para = para)
        Q_grids = range(0.0, 0.2, 301)
        B_grids = range(0.0, 2.0, 301)
        Vs = [V(B, Q) for B in B_grids, Q in Q_grids]
        push!(plts, contourf(Q_grids, B_grids, Vs, xlabel = "Q", ylabel = "B", title = "τ = $τ", c=:viridis, levels = 20, lw = 0))
    end
    plt = plot(plts..., layout = (2, 4), size = (3600, 1200))
    display(plt)
    savefig(plt, "./figure/V_tax.png")
end

begin
    τs = 0.075:0.01:0.315
    qs = similar(τs)
    ext_fin_B, ext_fin_Q = (df.B[ext_finance_date_id], df.Q[ext_finance_date_id]) .|> (x -> round(x, digits = 3))
    for (i, τ) in enumerate(τs)
        para = model(σ = res[1], α = res[2], ρ = res[3], τ = τ)
        V = solve_pde(para = para, damp = 0.5)
        qs[i] = para.q(V(ext_fin_B, ext_fin_Q))
    end
    plt = plot(τs, qs, title = "q($ext_fin_B, $ext_fin_Q; τ)", xlabel="τ", label="", c=:brown)
    display(plt)
    savefig(plt, "./figure/q_tax.png")
end

# simulate one path for mechanism explain 
begin
    x0 = [0.1, 0.0]
    τs = [0.075, 0.14]
    times = 0.0:0.1:50.0
    plts = []
    Q_ubs = [0.05, 0.3]
    B_ubs = [0.5, 5.0]
    for (i, τ) in enumerate(τs)
        para = model(σ = res[1], α = res[2], ρ = res[3], τ = τ)
        V = solve_pde(para = para)
        Q_grids = range(0.0, Q_ubs[i], 301)
        B_grids = range(0.0, B_ubs[i], 301)
        Vs = [V(B, Q) for B in B_grids, Q in Q_grids]
        plt = contourf(Q_grids, B_grids, Vs, xlabel = "Q", ylabel = "B", title = "τ = $τ", c=:viridis, levels = 20, lw = 0)
        sim = simulate(para; x0 = x0, n_sim = 1, tspan = (times[1], times[end]), seed = 302)
        path = hcat(only.(sim.(times))...)
        plot!(plt, path[2, :], path[1, :], label="", c=:brown, xlims = (Q_grids[1], Q_grids[end]), ylims = (B_grids[1], B_grids[end]), lw = 5)
        push!(plts, plt)
    end
    plt = plot(plts..., layout = (1, 2), size = (1600, 600))
    display(plt)
    savefig(plt, "./figure/mechanism.png")
end

# simulation for cutting benefit
begin
    times = 0.0:1/12:50.0
    cuts = 0.0:0.1:0.5
    n = length(cuts)
    p_benchmark = model().p
    l_benchmark = model().l
    plt = plot(title = "Bankruptcy Probability from Jan., 2020", xlabel = "year", leg=:outerright)
    for (i, cut) in enumerate(cuts)
        para = model(σ = res[1], α = res[2], ρ = res[3], p = (1-cut) * p_benchmark, l = (1-cut) * l_benchmark)
        sim = simulate(para, output_func = (sol, ctx) -> (sol.t[end], false), tspan=(0.0, times[end]))
        bankrupt_prob = [mean(sim.u .< t) for t in times]
        plot!(plt, times, bankrupt_prob, label="-$(100*cut)%", color = RGB(i/n, 0.2, 1-i/n))
    end
    display(plt)
    savefig(plt, "./figure/bankruptcy_prob_cuts.png")
end