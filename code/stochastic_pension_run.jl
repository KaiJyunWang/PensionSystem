using Plots, Roots, LogExpFunctions, LinearAlgebra
using DiffEqCallbacks, OrdinaryDiffEq, StochasticDiffEq
using Parameters, Printf, SteadyStateDiffEq
using DifferentialEquations, Interpolations
using Statistics, SparseArrays, ExponentialAction
using Optimization, OptimizationOptimJL, Random
using CSV, DataFrames, Dates

include("parameter_tables.jl")
using .ParameterTables: write_parameter_table

function model(; b = 0.00462, m = 0.058, T = 65, r = 0.041, p = 2.556, Tw = 20, Tm = 65,
    l = 20.61, τ = 0.115, y = 5.496, α = -7.752, β = 2.623, ρ = 0.02, σ = 0.083, B_max = 10.0)
    # find population growth rate 
    n = find_zero(n -> (-expm1(-n * Tm)) / n + exp(-n * Tm) / (n + m) - 1 / b, (-m+ 1e-8, -1e-8))

    # auxilary functions
    g(s) = s < Tm ? b * exp(-n * s) : b * exp(-n * s - m * (s - Tm)) 
    labor_force = b * ((exp(-n*Tw) - exp(-n*Tm))/n + exp(m*Tm) * (exp(-(m+n)*Tm) - exp(-(m+n)*T)) / (m+n))
    q(V) = V ≤ 0 ? logistic(-α / β) : logistic(-(α + l - V) / β)
    μB(B, Q, V) = (r-n) * B + b * τ * y * labor_force - (1 - q(V)) * g(T) * l - p * Q
    σB(B) = σ * B
    μQ(Q, V) = q(V) * g(T) - (m+n) * Q

    Q_max = b * exp(-n * T) / (m + n) 

    # precomputed differentiation matrix
    nB = 200
    nQ = 80
    Bs = range(0.0, B_max, nB)
    Qs = range(0.0, Q_max, nQ)

    dB = Bs[2] - Bs[1]
    dQ = Qs[2] - Qs[1]

    IB = sparse(I, nB, nB)
    IQ = sparse(I, nQ, nQ)

    fDB = spdiagm(0 => fill(-1/dB, nB), 1 => fill(1/dB, nB-1))
    bDB = spdiagm(0 => fill(1/dB, nB), -1 => fill(-1/dB, nB-1))
    DBB = spdiagm(0 => fill(-2/dB^2, nB), 1 => fill(1/dB^2, nB-1), -1 => fill(1/dB^2, nB-1))
    fDQ = spdiagm(0 => fill(-1/dQ, nQ), 1 => fill(1/dQ, nQ-1))
    bDQ = spdiagm(0 => fill(1/dQ, nQ), -1 => fill(-1/dQ, nQ-1))

    # adjust for boundary condition
    fDB[1, :] .= 0.0
    bDB[1, :] .= 0.0
    DBB[1, :] .= 0.0
    fDB[end, :] .= 0.0
    DBB[end, end] = -1/dB^2

    # kron for 2d grids 
    fDB = kron(IQ, fDB)
    bDB = kron(IQ, bDB)
    DBB = kron(IQ, DBB)
    fDQ = kron(fDQ, IB)
    bDQ = kron(bDQ, IB)

    return (; b, m, T, Tw, Tm, r, ρ, p, l, τ, y, n, α, β, σ, q, μB, σB, μQ, B_max, Q_max, 
        fDB, bDB, DBB, fDQ, bDQ, Bs, Qs)
end


para = model()

# helper function to return discretized infinitesimal generator
function construct_generator(V; para)
    @unpack μB, σB, μQ, fDB, bDB, DBB, fDQ, bDQ, Bs, Qs = para

    MμB = spdiagm(0 => [μB(B, Q, V[i,j]) for (i, B) in enumerate(Bs), (j, Q) in enumerate(Qs)] |> vec)
    MμQ = spdiagm(0 => [μQ(Q, V[i,j]) for (i, B) in enumerate(Bs), (j, Q) in enumerate(Qs)] |> vec)
    MσB = spdiagm(0 => [0.5 * σB(B)^2 for (i, B) in enumerate(Bs), (j, Q) in enumerate(Qs)] |> vec)

    return max.(MμB, 0) * fDB + min.(MμB, 0) * bDB + max.(MμQ, 0) * fDQ + min.(MμQ, 0) * bDQ + MσB * DBB
end

# solve the PDV of the pension
function solve_pde(; tol = 1e-8, iterations = 1000, para)
    @unpack m, r, ρ, p, μB, σB, μQ, B_max, Q_max, Bs, Qs = para

    V0 = [1/(ρ+m) * B / B_max for B in Bs, Q in Qs]
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
        V0 = V
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
function loss(θ; B_path = B_path, Q_path = Q_path, q_path = q_path, S = I, Δ = 1/12)
    para = model(; r = θ[1], σ = exp(θ[2]), α = θ[3], β = θ[4])
    V = solve_pde(; para = para)
    μB_targets = diff(B_path)./B_path[1:end-1]
     

    σ_target = std(μB_targets) / sqrt(Δ) - para.σ
    μB_mean = mean(μB_targets) - mean(Δ * para.μB.(B_path[1:end-1], Q_path[1:end-1], V.(B_path[1:end-1], Q_path[1:end-1])) ./ B_path[1:end-1])
    q_mean = para.q.(V.(B_path[1:end-1], Q_path[1:end-1])) - q_path[1:end-1]

    ms = vcat(μB_mean, σ_target, q_mean)
    return mean(abs2, S * ms)
end

# Estimation with real data 
df = CSV.read("data/labor_insurance_fund_monthly.csv", DataFrame)
df.monthly_pension_recipients[1:2] .= 0
df.lumpsum_recipients[1] = 0
plot(df.date, df.monthly_pension_recipients, label="monthly")
plot(df.date[1:end-1], diff(df.monthly_pension_recipients), label="monthly")
plot!(df.date, df.lumpsum_recipients, label="lumpsum")
q_path = diff(df.monthly_pension_recipients)./(diff(df.monthly_pension_recipients) + df.lumpsum_recipients[1:end-1])
plot(df.date[1:end-1], q_path, label="q")

ext_finance_date_id = findall(x -> x == Date(2020), df.date)|> only

df[!, :B] = df.fund_level_ntd ./ df.taiwan_registered_population ./ 100_000
df[!, :Q] = df.labor_insurance_old_age_pension_recipients ./ df.taiwan_registered_population
df[!, :q] = vcat(q_path, missing)
B_path = df.B[3:ext_finance_date_id-1]
Q_path = df.Q[3:ext_finance_date_id-1]
q_path = df.q[3:ext_finance_date_id-1] .|> Float64
X_path = collect.(zip(B_path, Q_path))

# plot the recipients 
begin
    p1 = plot(df.date[3:end], df.q[3:end], label="", c=:black, title="Proportion of Monthly")
    p2 = plot(df.date[3:end-1], diff(df.monthly_pension_recipients)[3:end], label="Monthly", leg=:topright)
    plot!(p2, df.date[3:end-1], df.lumpsum_recipients[3:end-1], label="Lumpsum", title="Number of New Recipients")
    vline!.([p1, p2], Ref([Date(2020)]), label="",  c=:black, ls=:dash)
    plt = plot(p1, p2, layout=(2,1), size=(800, 600))
    display(plt)
    savefig(plt, "./figure/recipients.png")
end

Δ = 1/12
begin
   p1 = plot(df.date[3:end-1], df.B[3:end-1], label = "", title="Fund Level per Capita (100,000 NTD)", c=:black)
   p2 = plot(df.date[3:end-1], df.Q[3:end-1], label = "", title="Recipients/Population", c=:black)
   #p3 = plot(df.date[3:end-1], df.q[3:end-1], title="q", label="", c=:black)
   vline!.([p1, p2], Ref([Date(2020)]), label="",  c=:black, ls=:dash)
   plt = plot(p1, p2, layout = (2, 1), size = (800, 600))
   display(plt)
   savefig(plt, "./figure/state_data.png")
end

# initial guesses
r0 = 0.02
σ0 = 0.1
α0 = -3.0
β0 = 2.0
θ0 = [r0, log(σ0), α0, β0]

optf = OptimizationFunction((θ, p) -> loss(θ), AutoFiniteDiff())
prob = OptimizationProblem(optf, θ0)
sol = solve(prob, NelderMead(); show_trace = true)
res = sol.u |> (x -> [x[1], exp(x[2]), x[3], x[4]])

para_est = model(; r = res[1], σ = res[2], α = res[3], β = res[4])

# output parameter table
write_parameter_table(para_est)
write_parameter_table(para_est; output_path = "table/parameters_A.tex", panel = :A)
write_parameter_table(para_est; output_path = "table/parameters_B.tex", panel = :B)


# simulate the estimated model
V_est = solve_pde(para = para_est, iterations = 1000)

Q_grids = range(0.0, para.Q_max, 301)
B_grids = range(0.0, para.B_max, 301)
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
    plt = plot(df.date, df.q, title="q", label="Data", c=:black)
    vline!(plt, [Date(2020)], label="",  c=:black, ls=:dash)    
    plot!(plt, df.date, para_est.q.(V_est.(df.B, df.Q)), label="Model", c=:brown)
    display(plt)
    savefig(plt, "./figure/validation_q.png")
end

# simulation on the estimated parameters
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

sde_prob = SDEProblem(drift_est!, diffusion_est!, [df.B[3], df.Q[3]], (0.0, Δ * (length(df.B[3:end])-1)), para_est)
n_sim = 20
Random.seed!(21)
seeds = rand(UInt64, n_sim)
ensemble_prob = EnsembleProblem(
        sde_prob, 
        prob_func = (prob, ctx) -> remake(prob; seed = seeds[ctx.sim_id])
    )
sol = solve(
        ensemble_prob,
        SRIW1(),
        EnsembleThreads();
        trajectories = n_sim, 
        callback = cb, 
        abstol = 1e-8
    )
# sim = solve(sde_prob, SRIW1(), callback = cb, abstol = 1e-8, seed = 1234)

# plot simulation
begin
    times = 0.0:Δ:(length(df.date)-1)*Δ
    dates = df.date
    p1 = plot(dates, df.B, label = "Data", title="B", c=:black)
    p2 = plot(dates, df.Q, label = "Data", title="Q", c=:black)
    p3 = plot(dates, df.q, title="q", label = "Data", c=:black)
    vline!.([p1, p2, p3], Ref([Date(2020)]) , c=:black, label="", ls=:dash)
    
    sim_Bs = hcat([getindex.(sol(t), 1) for t in times]...)
    sim_Qs = hcat([getindex.(sol(t), 2) for t in times]...)
    sim_qs = para_est.q.(V_est.(sim_Bs, sim_Qs))
    for i in 1:n_sim 
        lab = i==1 ? "Simulation" : ""
        plot!(p1, dates, sim_Bs[i, :], label=lab, c=:brown)
        plot!(p2, dates, sim_Qs[i, :], label=lab, c=:brown)
        plot!(p3, dates, sim_qs[i, :], label=lab, c=:brown)
    end
    plt = plot(p1, p2, p3, layout = (3, 1), size = (800, 1000))
    display(plt)
    savefig(plt, "./figure/estimated_sim.png")
end

# simulation with no external finance 
sim_horizon = times[findfirst(x->x==Date(2020), df.date[3:end])-1:end]
sde_prob = SDEProblem(drift_est!, diffusion_est!, [df.B[ext_finance_date_id-1], df.Q[ext_finance_date_id-1]], (sim_horizon[1], sim_horizon[end]), para_est)
n_sim = 20
Random.seed!(2026)
seeds = rand(UInt64, n_sim)
ensemble_prob = EnsembleProblem(
        sde_prob, 
        prob_func = (prob, ctx) -> remake(prob; seed = seeds[ctx.sim_id])
    )
sol = solve(
        ensemble_prob,
        SRIW1(),
        EnsembleThreads();
        trajectories = n_sim, 
        callback = cb, 
        abstol = 1e-8
    )

sim_Bs = hcat([getindex.(sol(t), 1) for t in sim_horizon]...)
sim_Qs = hcat([getindex.(sol(t), 2) for t in sim_horizon]...)
sim_qs = para_est.q.(V_est.(sim_Bs, sim_Qs))

begin
    times = 0.0:Δ:(length(df.B[3:end])-1) * Δ
    dates = df.date[findfirst(x->x==Date(2020), df.date[3:end])-1:end]
    p1 = plot(df.date, df.B, label = "Data", title="B", c=:black)
    p2 = plot(df.date, df.Q, label = "Data", title="Q", c=:black)
    p3 = plot(df.date, df.q, title="q", label = "Data", c=:black)

    vline!.([p1, p2, p3], Ref([Date(2020)]) , c=:black, ls=:dash, label="")

    for i in 1:n_sim 
        lab = i==1 ? "Simulation" : ""
        plot!(p1, dates, sim_Bs[i, :], label=lab, c=:brown)
        plot!(p2, dates, sim_Qs[i, :], label=lab, c=:brown)
        plot!(p3, dates, sim_qs[i, :], label=lab, c=:brown)
    end
    plt = plot(p1, p2, p3, layout = (3, 1), size = (800, 1000))
    display(plt)
    savefig(plt, "./figure/no_ext_finance.png")
end

# simulate the case for the case everyone choose lumpsum
para_est = model(; r = res[1], σ = res[2], α = Inf, β = res[4])
V_est = solve_pde(para = para_est, iterations = 1000)

Q_grids = range(0.0, para.Q_max, 301)
B_grids = range(0.0, para.B_max, 301)
V_ests = [V_est(B, Q) for B in B_grids, Q in Q_grids]
begin
    p1 = surface(Q_grids, B_grids, V_ests, xlabel = "Q", ylabel = "B", title = "V", camera = (50,30), alpha = 0.7, c=:viridis)
    p2 = contourf(Q_grids, B_grids, V_ests, xlabel = "Q", ylabel = "B", title = "V", c=:viridis, levels = 20, lw = 0)
    plt = plot(p1, p2, layout=(1, 2), size=(800, 400))
    display(plt)
    savefig(plt, "./figure/lumpsum_V.png")
end
# simulation on the estimated parameters
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

sde_prob = SDEProblem(drift_est!, diffusion_est!, [df.B[3], df.Q[3]], (0.0, Δ * (length(df.B[3:end])-1)), para_est)
n_sim = 20
Random.seed!(21)
seeds = rand(UInt64, n_sim)
ensemble_prob = EnsembleProblem(
        sde_prob, 
        prob_func = (prob, ctx) -> remake(prob; seed = seeds[ctx.sim_id])
    )
sol = solve(
        ensemble_prob,
        SRIW1(),
        EnsembleThreads();
        trajectories = n_sim, 
        callback = cb, 
        abstol = 1e-8
    )
# sim = solve(sde_prob, SRIW1(), callback = cb, abstol = 1e-8, seed = 1234)

# plot simulation
begin
    times = 0.0:Δ:(length(df.date)-1)*Δ
    dates = df.date
    plt = plot(dates, df.B, label = "Data", title="B", c=:black, ylims=(0.0, 0.8))
    vline!(plt, [Date(2020)] , c=:black, label="", ls=:dash)
    
    sim_Bs = hcat([getindex.(sol(t), 1) for t in times]...)
    for i in 1:n_sim 
        lab = i==1 ? "Simulation" : ""
        plot!(plt, dates, sim_Bs[i, :], label=lab, c=:brown)
    end
    display(plt)
    savefig(plt, "./figure/lumpsum_sim.png")
end

