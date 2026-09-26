using DifferentialEquations, StochasticDiffEq
using DiffEqCallbacks, OrdinaryDiffEq
using Plots, Roots, LogExpFunctions
using Parameters, Printf

# solve the population equation up to t_end
# t_end, a_grids, g = include("population.jl")

# monetary unit: 100,000 
# Need to reconstruct the model parameters to reset Tr each iteration. 
# β for the scaling parameter in the T1EV residual  
function model(; b = 0.00462, m = 0.058, T = 65, r = 0.04, p = 2.556, 
    l = 20.61, τ = 0.125, y = 0.458, α = -6.0, β = 1.0)
    # find population growth rate 
    n = find_zero(n -> (-expm1(-n * T)) / n + exp(-n * T) / (n + m) - 1 / b, (-m+ 1e-8, -1e-8))

    return (; b, m, T, r, p, l, τ, y, n, α, β)
end

function q(Tr; para)
    @unpack b, m, T, r, p, l, τ, y, n, α, β = para
    return logistic(-(α + l - p * (-expm1(-(r+m) * Tr)) / (r+m)) / β)
end

# drift term of the ODE system 
# u = [B, Q, N, Tr]
function drift!(du, u, para, t)
    @unpack b, m, T, r, p, l, τ, y, n, α, β = para

    du[1] = r * u[1] + u[3] * b * τ * y / n * (1 - exp(-n * T)) - (1 - q(u[4]; para = para)) * b * exp(-n * T) * u[3] * l - p * u[2] 
    du[2] = q(u[4]; para = para) * b * exp(-n * T) * u[3] - m * u[2]
    du[3] = n * u[3]
    du[4] = -1.0 
end

function solve_Tr(; damp = 0.5, tol = 1e-7, iterations = 1000, 
        Tr0 = 11.0, u0 = [3.0, 0.0, 1.0, Tr0], tspan = (0.0, 50.0), 
        alg = Tsit5(), para)

    # define callback for the event B = 0
    function condition(u, t, integrator) # Event when condition(u,t,integrator) == 0
        u[1]
    end
    cb = ContinuousCallback(condition, terminate!)

    iter = 0
    resid = Inf 

    while resid > tol && iter < iterations 
        prob = ODEProblem(drift!, vcat(u0[1:3], Tr0), tspan, para)
        sol = solve(prob, alg, callback = cb)
        resid = abs(Tr0 - sol.t[end])
        iter += 1
        if iter % 10 == 0 || resid < tol
            @printf "Iteration %d \t Tr = %.6f \t Rel. Error = %.6e\n" iter Tr0 resid
        end
        # update the Tr guess with damping 
        Tr0 = sol.t[end] * damp + Tr0 * (1 - damp)
    end

    # return the solution to the model 
    prob = ODEProblem(drift!, vcat(u0[1:3], Tr0), tspan, para)
    sol = solve(prob, alg, 
        callback = ContinuousCallback(
            (u, t, integrator) -> u[1], 
            nothing, 
            terminate!
            )
        )

    return sol, Tr0
end

para = model()
sol, Tr0 = solve_Tr(; para = para)

begin
    ts = range(0.0, stop = sol.t[end], length = 500)
    p1 = plot(sol, idxs = 1, title = "B", xlabel = "t", label = "")
    p2 = plot(sol, idxs = 2, title = "Q", xlabel = "t", label = "")
    p3 = plot(ts, q.(reverse(ts); para = para), title = "q", xlabel = "t", label = "")
    p4 = plot(sol, idxs = 4, title = "Ref", xlabel = "t", label = "")
    plot!(p1, sol.t, zeros(size(sol.t)), label = "B_bar", linestyle = :dash, c = :brown)
    plt = plot(p1, p2, p3, p4, layout = (4, 1), size = (600, 1000))
end
