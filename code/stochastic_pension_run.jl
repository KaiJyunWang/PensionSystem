using Plots, Roots, LogExpFunctions, LinearAlgebra
using DiffEqCallbacks, OrdinaryDiffEq, StochasticDiffEq
using Parameters, Printf, SteadyStateDiffEq
using DifferentialEquations, FastChebInterp
using Statistics

include("collocation_diff_matrix.jl")

function model(; b = 0.00462, m = 0.058, T = 65, r = 0.03, p = 2.556, 
    l = 20.61, τ = 0.125, y = 5.496, α = -12.0, β = 2.0, σ = 0.1, B_max = 5.0)
    # find population growth rate 
    n = find_zero(n -> (-expm1(-n * T)) / n + exp(-n * T) / (n + m) - 1 / b, (-m+ 1e-8, -1e-8))

    # auxilary functions
    q(V) = V == 0 ? logistic(-α / β) : logistic(-(α + l - p * max(V, 0)) / β)
    μB(B, Q, V) = (r-n) * B + b * τ * y / n * (exp(-n * 20)-exp(-n * T)) - (1 - q(V)) * b * exp(-n * T) * l - p * Q
    σB(B) = σ * B
    μQ(Q, V) = q(V) * b * exp(-n * T) - (m+n) * Q

    Q_max = b * exp(-n * T) / (m + n) 

    return (; b, m, T, r, p, l, τ, y, n, α, β, σ, q, μB, σB, μQ, B_max, Q_max)
end

para = model()

# solve the PDV of the pension
function solve_pde(; tol = 1e-8, iterations = 1000, para, ns = (100, 20))
    @unpack b, m, T, r, p, l, τ, y, n, α, β, σ, q, μB, σB, μQ, B_max, Q_max = para

    lbs = [0.0, 0.0]
    ubs = [B_max, Q_max]

    # collocation points
    xs = chebpoints(ns, lbs, ubs)
    Bs = [x[1] for x in xs]
    Qs = [x[2] for x in xs]

    DB, DQ = collocation_diff_matrix(xs, lbs, ubs)
    DB2 = DB^2

    V0 = [1 / (r + m) * x[1] / B_max for x in xs]

    upper_B_bc_positions = 1:(ns[1]+1):prod(ns.+1)
    lower_B_bc_positions = ns[1]:(ns[1]+1):prod(ns.+1)

    id_mat = Matrix(I, prod(ns.+1), prod(ns.+1))

    rhs = ones(prod(size(V0)))
    rhs[upper_B_bc_positions] .= 0.0#1/(r+m)
    rhs[lower_B_bc_positions] .= 0.0

    iter = 1 
    rel_error = Inf
    while iter ≤ iterations && rel_error > tol
        MμB = Diagonal(vec(μB.(Bs, Qs, V0)))
        MμQ = Diagonal(vec(μQ.(Qs, V0)))
        MσB = Diagonal(vec(σB.(Bs)))
        A = (r + m) * I - MμB * DB - MμQ * DQ - 0.5 * MσB * DB2
        # for boundary condition 
        #A[upper_B_bc_positions, :] = id_mat[upper_B_bc_positions, :]
        A[upper_B_bc_positions, :] = DB[upper_B_bc_positions, :]
        A[lower_B_bc_positions, :] = id_mat[lower_B_bc_positions, :]

        V = (A \ rhs) |> (x -> reshape(x, ns[1]+1, ns[2]+1)) 
        #V[1, :] .= 1/(r+m)
        V[end, :] .= 0.0
        rel_error = maximum(abs, V - V0)
        if iter % iterations == 0
            @printf "Iterations: %d \t Rel. Error: %.5g \n" iter rel_error
        end
        iter += 1
        V0 = V
    end

    inner_V = chebinterp(V0, lbs, ubs)

    # wrap up the solution 
    return (B, Q) -> begin
        inner_V(clamp.([B, Q], (0.0, 0.0), (B_max, Q_max)))
    end
end

V = solve_pde(; para = para)

Q_grids = range(0.0, para.Q_max, 101)
B_grids = range(0.0, para.B_max, 101)
Vs = [V(B, Q) for B in B_grids, Q in Q_grids]
surface(Q_grids, B_grids, Vs, xlabel = "Q", ylabel = "B", title = "V", camera = (50,30), alpha = 0.7)
contourf(Q_grids, B_grids, Vs, xlabel = "Q", ylabel = "B", title = "V")

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

sde_prob = SDEProblem(drift!, diffusion!, [0.03, 0.0], (0.0, 15.0), para)
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


# solve for the forward equation using finite volume method
function construct_generator(V; para, nB = 100, nQ = 80)
    @unpack b, m, T, r, p, l, τ, y, n, α, β, σ, q, μB, σB, μQ, B_max, Q_max = para
    A_size = nB * nQ
    Bs = range(0.0, B_max, nB)
    Qs = range(0.0, Q_max, nQ)
    dB = Bs[2] - Bs[1]
    dQ = Qs[2] - Qs[1]

    main_diag = zeros(A_size) 
    upper_diagB = -[min(μB(B+0.5*dB, Q, V(B+0.5*dB, Q)), 0)/dB - 0.5 * σ^2 * (B+dB)^2 / dB^2 for B in Bs, Q in Qs] |> vec
    lower_diagB = -[-max(μB(B-0.5*dB, Q, V(B-0.5*dB, Q)), 0)/dB - 0.5 * σ^2 * (B-dB)^2 / dB^2 for B in Bs, Q in Qs] |> vec
    upper_diagQ = -[min(μQ(Q+0.5*dQ, V(B, Q+0.5*dQ)), 0)/dQ for B in Bs, Q in Qs[2:end]] |> vec
    lower_diagQ = -[-max(μQ(Q-0.5*dQ, V(B, Q-0.5*dQ)), 0)/dQ for B in Bs, Q in Qs[1:end-1]] |> vec
    
    
    # modify for boundaries
    upper_diagB[nB:nB:A_size] .= 0
    lower_diagB[1:nB:A_size] .= 0

    A = spdiagm(0 => main_diag, 1 => upper_diagB[1:end-1], -1 => lower_diagB[2:end], nB => upper_diagQ, -nB => lower_diagQ)
    
    row_sums = A * ones(A_size)
    A[diagind(A)] .= -row_sums
    # return forward generator. Can obtain the solution by exp(tA)m0
    return Bs, Qs, A
end


#---------------------------------------
# Estimation

using Optimization, OptimizationOptimJL, SparseArrays, ExponentialAction

# generated data
#=
times = 0.0:1/12:15.0 |> collect
Xs = sim.(times)

B_path = getindex.(Xs, 1)
Q_path = getindex.(Xs, 2)
=#

# method of moments with Euler-Maruyama type approximation 
function loss(θ; B_path = B_path, Q_path = Q_path, W = I, Δ = 1/12)
    para = model(; σ = exp(θ[1]), α = θ[2], β = exp(θ[3]))
    V = solve_pde(; para = para)

    q_targets = ((Q_path[2:end] - Q_path[1:end-1])/Δ + (para.m + para.n) * Q_path[1:end-1]) / (para.b * exp(-para.n * para.T))
    qs = [para.q(V(B, Q)) for (B, Q) in zip(B_path[1:end-1], Q_path[1:end-1])]
    σ_target = std((diff(B_path) - para.μB.(B_path[1:end-1], Q_path[1:end-1], V.(B_path[1:end-1], Q_path[1:end-1])) * Δ) ./ (B_path[1:end-1] * sqrt(Δ)))
    mean_μB_target = mean(diff(B_path) ./ (B_path[1:end-1] * Δ))
    mean_μB = mean([para.μB(B, Q, V(B, Q))/B for (B, Q) in zip(B_path[1:end-1], Q_path[1:end-1])])

    ms = vcat(qs - q_targets, para.σ - σ_target, mean_μB - mean_μB_target)
    return ms' * W * ms
end

#=
θ_true = [log(para.r), log(para.σ), para.α, log(para.β)]
θ0 = θ_true + randn(4); θ0 |> (x -> [exp(x[1]), exp(x[2]), x[3], exp(x[4])])

optf = OptimizationFunction((θ, p) -> loss(θ), AutoFiniteDiff())
prob = OptimizationProblem(optf, θ0)
sol = solve(prob, NelderMead(); show_trace=true)
res = sol.u |> (x -> [exp(x[1]), exp(x[2]), x[3], exp(x[4])])

para_est = model(; r = res[1], σ = res[2], α = res[3], β = res[4])
V_est = solve_pde(; para = para_est)

Q_grids = range(0.0, para.Q_max, 100)
B_grids = range(0.0, 30, 100)
Vs = [V(B, Q) for B in B_grids, Q in Q_grids]
V_ests = [V_est(B, Q) for B in B_grids, Q in Q_grids]
surface(Q_grids, B_grids, V_ests, xlabel = "Q", ylabel = "B", title = "V_est", camera = (50,30), alpha = 0.7)
=#

# Estimation with real data 
using CSV, DataFrames

df = CSV.read("data/labor_insurance_fund_monthly.csv", DataFrame)

df[!, :B] = df.fund_level_ntd ./ df.taiwan_registered_population ./ 100_000
df[!, :Q] = df.labor_insurance_old_age_pension_recipients ./ df.taiwan_registered_population

Δ = 1/12
using Dates
begin
   p1 = plot(df.date[3:end-1], df.B[3:end-1], label = "", title="B")
   p2 = plot(df.date[3:end-1], df.Q[3:end-1], label = "", title="Q")
   p3 = plot(df.date[3:end-1], diff(df.Q[3:end])/Δ/(para.b * exp(-para.n*para.T)), title="q", label="")
   vline!.([p1, p2, p3], Ref([Date(2020)]), label="",  c=:brown)
   plt = plot(p1, p2, p3, layout = (3, 1), size = (600, 1000))
end

# initial guesses
ext_finance_date_id = findall(x -> x == Date(2020), df.date)|> only

σ0 = 0.1
α0 = -18.0
β0 = 2.0
θ0 = [log(σ0), α0, log(β0)]

optf = OptimizationFunction((θ, p) -> loss(θ; B_path = df.B[3:ext_finance_date_id-1], Q_path = df.Q[3:ext_finance_date_id-1]), AutoFiniteDiff())
prob = OptimizationProblem(optf, θ0)
sol = solve(prob, NelderMead(); show_trace=true)
res = sol.u |> (x -> [exp(x[1]), x[2], exp(x[3])])

para_est = model(; σ = res[1], α = res[2], β = res[3])
V_est = solve_pde(para = para_est, iterations = 1000)

Q_grids = range(0.0, para.Q_max, 101)
B_grids = range(0.0, para.B_max, 101)
V_ests = [V_est(B, Q) for B in B_grids, Q in Q_grids]
surface(Q_grids, B_grids, V_ests, xlabel = "Q", ylabel = "B", title = "V", camera = (50,30), alpha = 0.7, c=:viridis)
contourf(Q_grids, B_grids, V_ests, xlabel = "Q", ylabel = "B", title = "V", c=:thermal)

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

sde_prob = SDEProblem(drift_est!, diffusion_est!, [df.B[3], df.Q[3]], (0.0, 15.0), para_est)
sim = solve(sde_prob, SRIW1(), callback = cb, abstol = 1e-8)

# plot simulation
begin
    times = 0.0:1/12:(length(df.B[3:end])-1)/12
    p1 = plot(times, df.B[3:end], label = "data", title="B", c=:brown)
    p2 = plot(times, df.Q[3:end], label = "data", title="Q", c=:brown)
    ts = range(0.0, sim.t[end], 300)
    #sim_Vs = map(u -> V(u...), sim.(ts))
    plot!(p1, sim, idxs = 1, title = "B", xlabel = "t", label = "sim")
    plot!(p1, sim.t, zeros(length(sim.t)), label = "", c=:black, ls=:dash)
    plot!(p2, sim, idxs = 2, title = "Q", xlabel = "t", label = "sim")
    #p3 = plot(ts, para.q.(sim_Vs), title = "q", xlabel = "t", label = "", xlims = (0.0, sim.t[end]))
    #p4 = plot(ts, sim_Vs, title = "V", xlabel = "t", label = "", xlims = (0.0, sim.t[end]))
    plt = plot(p1, p2, layout = (2, 1), size = (600, 1000))
end