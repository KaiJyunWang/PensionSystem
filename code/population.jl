using Interpolations, Plots, Roots

raw_survival = parse.(Float64, readlines("data/survival_curve.txt"))
pop_2009 = parse.(Float64, readlines("data/age_dist_2009.txt"))
birth_counts = parse.(Float64, readlines("data/birth_counts.txt"))

# normalize initial mass to 1
raw_survival = raw_survival ./ raw_survival[1]

# interpolation and extrapolate linearly from raw curve
itp_S = extrapolate(interpolate(0:length(raw_survival)-1, raw_survival, SteffenMonotonicInterpolation()), Line())
age_max = find_zero(a -> itp_S(a), 100)
itp_S2(a) = a < age_max ? itp_S(a) : 0

# distribute the last entry (100+) to 100:floor(age_max) equally
pop_2009_norm = pop_2009 ./ sum(pop_2009)
pop_2009_norm = vcat(pop_2009_norm[1:end-1], pop_2009_norm[end]/(floor(Int, age_max)-100)*ones(floor(Int, age_max)-100), 0)

# interpolate initial condition 
itp_g0 = extrapolate(interpolate((collect(0:floor(Int, age_max)),), pop_2009_norm, Gridded(Constant{Previous}())), 0)

# obtain fertility curve 
fertility_ages = 15:49
fertility_curve_data = birth_counts ./ pop_2009[fertility_ages .+ 1]

itp_b = extrapolate(interpolate((fertility_ages,), fertility_curve_data, Gridded(Linear())), 0)

# solve volterra equation with time discretization dt
function solve_volterra(; g0 = itp_g0, n_t = 300, t_end = 100, b = itp_b, S = itp_S2, age_max = age_max)
    t_grids = range(0, t_end, n_t)
    dt = t_grids[2] - t_grids[1]

    a_grids = range(0, age_max, 200) |> collect
    da = a_grids[2] - a_grids[1]

    f = zeros(length(t_grids))

    for (k, t) in enumerate(t_grids)
        if k == 1 
            f[k] = sum(b.(t_grids) .* g0.(t_grids)) * dt
        else
            f[k] = dt * sum(b.(t .- t_grids[1:k-1]) .* S.(t .- t_grids[1:k-1]) .* f[1:k-1]) + da * sum(b.(t .+ a_grids[1:end-1]) .* g0.(a_grids[1:end-1]) .* S.(t .+ a_grids[1:end-1]) ./ S.(a_grids[1:end-1]))
        end
    end
    return interpolate((collect(t_grids),), f, Gridded(Linear()))
end

function solution_wrapper(f; g0 = itp_g0, S = itp_S2)
    return function g(t, s)
        if s > t 
            return g0(s-t) * S(s) / S(s-t)
        else
            return f(t-s) * S(s)
        end
    end
end

g = solution_wrapper(solve_volterra())

ages = range(0, age_max, 200)

begin
    t_end = 100
    plt = plot()
    for t in 0:50:t_end
        plot!(ages, g.(t, ages), label="", c=RGB(t/t_end, 0.2, 1-t/t_end))
    end
    plt
end