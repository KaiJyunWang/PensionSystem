using DifferentialEquations, StochasticDiffEq
using DiffEqCallbacks, OrdinaryDiffEq
using Plots, Parameters, Statistics

para = (; r = 0.05, c = 0.1, σ = 0.5)

function drift!(du, u, p, t)
    @unpack r, c, σ = p
    du[1] = r * u[1] - c 
end

function diffusion!(du, u, p, t)
    @unpack r, c, σ = p
    du[1] = σ * u[1]
end

tspan = (0.0, 40.0)

cb = ContinuousCallback(
    (u, t, integrator) -> u[1],  
    nothing,
    (integrator) -> begin
        terminate!(integrator)
    end, 
    save_positions = (true, true)
)

prob = SDEProblem(drift!, diffusion!, [1.0], tspan, para)


# ensemble simulation
n_sim = 10_000
output_func(sol, ctx) = (sol.t[end], false)
seeds = rand(UInt64, n_sim)
prob_func(prob, ctx, repeat) = remake(prob; seed = seeds[ctx])

ensemble_prob = EnsembleProblem(prob; output_func = output_func)

sim = solve(
    ensemble_prob,
    SRIW1(),
    EnsembleThreads();
    trajectories = n_sim,
    callback = cb,
    save_everystep = false
)

sim.u

histogram(sim.u, normalize=:pdf, color=:gray, bins = range(0.0, 40.0, 30), label = "p_T")

mean(x -> x > 39.99, sim.u)

mean(sim.u)
