# Continued zero-reserve pension system: endogenous payroll tax funds both payment types.
# Run separately from the other pension scenario.
using LinearAlgebra, SparseArrays
using Interpolations
using OrdinaryDiffEq, SteadyStateDiffEq
using CairoMakie

include(joinpath(@__DIR__, "extension_benefits.jl"))

# Monetary unit: NT$1,000,000. Ages and rates are measured in years.
const raw_mortality = parse.(Float64, readlines(joinpath(@__DIR__, "..", "..", "data", "mortality.txt")))
const log_mortality = extrapolate(
    interpolate((0:length(raw_mortality)-1,), log.(raw_mortality), Gridded(Linear())), Line())

function model(; b = 0.00462, γ = 2.0, ρ = 0.01,
    θb = 1.0, κ = 3.0, A = 1.0, α = 0.33,
    ψ = s -> max(0.0, s - 60),
    τ0 = 0.12, tax_cap = 0.550,
    p = 0.256, l = 2.061,
    Tw = 20, Tr = 65, Tr_lb = Tr - 5, Tr_ub = Tr + 5,
    benefit_adjustment = 0.04,
    m = s -> exp(log_mortality(s)), T = 100,
    z_bar = s -> -1.656 + 0.052s - 0.000573s^2,
    z_vals = [-Inf, -0.451, 0.197, 0.0, 0.232, 0.653],
    Λ = Tridiagonal(fill(0.1, 5), [-0.2, -0.3, -0.3, -0.3, -0.3, -0.1], fill(0.2, 5)))

    τ = y -> τ0 * min(y, tax_cap)
    return (; b, γ, θb, κ, ρ, A, α, ψ, τ0, tax_cap, τ, p, l, Tw, Tr, Tr_lb, Tr_ub, benefit_adjustment,
        m, T, z_bar, z_vals, Λ)
end

# Use the off-diagonal entries of Λ as jump rates and reconstruct its
# diagonal so each row of the generator sums to zero.
function productivity_generator(mp)
    nz = length(mp.z_vals)
    size(mp.Λ) == (nz, nz) || throw(ArgumentError("Λ must match z_vals"))
    Q = Matrix{Float64}(mp.Λ)
    for i in 1:nz
        Q[i, i] = 0.0
        any(Q[i, :] .< 0) && throw(ArgumentError("Λ has a negative jump rate"))
        Q[i, i] = -sum(Q[i, :])
    end
    return Q
end

function invariant_distribution(Q)
    M = copy(transpose(Q))
    M[end, :] .= 1.0
    rhs = zeros(size(Q, 1)); rhs[end] = 1.0
    return M \ rhs
end

utility(c, γ) = γ == 1 ? log(c) : c^(1 - γ) / (1 - γ)
bequest(a, mp) = mp.θb * utility(a + mp.κ, mp.γ)

# Refine existing age intervals inside the retirement window without changing
# the rest of the base grid or creating nearly duplicate boundary points.
function age_grid(mp, n_age, retirement_age_refinement)
    n_age >= 3 || throw(ArgumentError("n_age must be at least 3"))
    retirement_age_refinement isa Integer && retirement_age_refinement >= 1 ||
        throw(ArgumentError("retirement_age_refinement must be a positive integer"))
    base = collect(range(0.0, mp.T, length = n_age))
    indices = Int[]
    for boundary in (mp.Tw, mp.Tr_lb, mp.Tr_ub)
        index = findfirst(age -> isapprox(boundary, age; atol = 1e-10), base)
        index === nothing && throw(ArgumentError(
            "Tw, Tr_lb and Tr_ub must be on the base age grid"))
        base[index] = boundary
        push!(indices, index)
    end
    ages = Float64[base[1]]
    for i in 1:length(base)-1
        substeps = indices[2] <= i < indices[3] ? retirement_age_refinement : 1
        append!(ages, range(base[i], base[i+1], length = substeps + 1)[2:end])
    end
    return ages
end

function age_population(mp, ages)
    m = mp.m.(ages)
    h = diff(ages)
    hazard = cumsum(vcat(0.0, h .* (m[1:end-1] .+ m[2:end]) ./ 2))
    weights = vcat(h[1]/2, (h[1:end-1] .+ h[2:end])/2, h[end]/2)
    density(η) = mp.b .* exp.(-η .* ages .- hazard)
    residual(η) = dot(weights, density(η)) - 1
    lo, hi = -1.0, 1.0
    residual(lo) > 0 && residual(hi) < 0 || error("Could not bracket population growth")
    for _ in 1:80
        mid = (lo + hi)/2
        if residual(mid) > 0
            lo = mid
        else
            hi = mid
        end
    end
    η = (lo + hi)/2
    return (; η, density = density(η), mortality = m, weights)
end

# Split the asset grid at a dense-state threshold. The lower segment gets most
# intervals; the sparse upper segment preserves the state constraint at a_max.
function asset_grid(n_assets, a_max, dense_asset_fraction, dense_grid_share)
    0 < dense_asset_fraction < 1 ||
        throw(ArgumentError("dense_asset_fraction must be between 0 and 1"))
    0 < dense_grid_share < 1 ||
        throw(ArgumentError("dense_grid_share must be between 0 and 1"))
    n_intervals = n_assets - 1
    n_dense = clamp(round(Int, dense_grid_share*n_intervals), 1, n_intervals - 1)
    cutoff = dense_asset_fraction*a_max
    dense = collect(range(0.0, cutoff, length = n_dense + 1))
    tail = collect(range(cutoff, a_max, length = n_intervals - n_dense + 1))
    return vcat(dense, tail[2:end])
end

# Retirees have enough extra grid space to receive l at the largest worker
# asset. Linear interpolation and its transpose preserve the asset jump.
function retirement_asset_grid(assets, l)
    l == 0 && return copy(assets)
    ntail = max(1, ceil(Int, l/(assets[end] - assets[end-1])))
    return vcat(assets, collect(range(assets[end], assets[end] + l,
        length = ntail + 1))[2:end])
end

function lump_sum_operator(assets, retired_assets, l)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for j in eachindex(assets)
        target = assets[j] + l
        lo = min(searchsortedlast(retired_assets, target), length(retired_assets)-1)
        hi = lo + 1
        weight = (target - retired_assets[lo])/(retired_assets[hi] - retired_assets[lo])
        push!(rows, j, j); push!(cols, lo, hi); push!(vals, 1-weight, weight)
    end
    return sparse(rows, cols, vals, length(assets), length(retired_assets))
end

pension_tax(mp, tax_rate, earnings) = tax_rate*min(earnings, mp.tax_cap)

function asset_cell_widths(assets)
    return vcat((assets[2] - assets[1])/2,
        (assets[3:end] .- assets[1:end-2])./2,
        (assets[end] - assets[end-1])/2)
end

# Conservative generator for first-order upwind differences on an asset grid.
# Rows sum to zero; its transpose is the Kolmogorov forward operator.
function asset_generator(drift, assets)
    na, nz = size(drift)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for z in 1:nz, a in 1:na
        i = a + (z-1)*na
        d = drift[a, z]
        if d > 0 && a < na
            rate = d/(assets[a+1] - assets[a])
            push!(rows, i, i); push!(cols, i, i+1); push!(vals, -rate, rate)
        elseif d < 0 && a > 1
            rate = -d/(assets[a] - assets[a-1])
            push!(rows, i, i); push!(cols, i, i-1); push!(vals, -rate, rate)
        end
    end
    return sparse(rows, cols, vals, na*nz, na*nz)
end

function worker_generator(drift, assets, Q)
    na, nz = size(drift)
    return asset_generator(drift, assets) + kron(sparse(Q), sparse(I, na, na))
end

# CUDA sparse QR is available for explicit GPU runs. The current 1,440-state
# system is faster with CPU sparse LU, so GPU loading is deferred until requested.
const gpu_module = Ref{Union{Nothing, Module}}(nothing)

function cuda_sparse_solve(M, rhs)
    cuda = gpu_module[]::Module
    dM = getfield(cuda, :CuSparseMatrixCSR)(M)
    db = getfield(cuda, :CuArray)(rhs)
    dx = similar(db)
    getfield(getfield(cuda, :cuSOLVER), :csrlsvqr!)(
        dM, db, dx, 1e-12, Cint(1), 'O')
    return Array(dx)
end

function sparse_system_solve(M, rhs, backend)
    backend === :cpu && return M \ rhs
    if gpu_module[] === nothing
        cuda = Base.require(Base.PkgId(
            Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        Base.invokelatest(getfield(cuda, :functional)) ||
            error("CUDA GPU is not available")
        gpu_module[] = cuda
    end
    if rhs isa AbstractMatrix
        return hcat([Base.invokelatest(cuda_sparse_solve, M, rhs[:, j])
            for j in axes(rhs, 2)]...)
    end
    return Base.invokelatest(cuda_sparse_solve, M, rhs)
end

# The HJB chooses a forward or backward slope according to its asset drift.
# Zero drift enforces the borrowing and upper asset-grid constraints.
function consumption_policy(V, assets, r, income, γ)
    na, nz = size(V)
    c = similar(V)
    drift = similar(V)
    for z in 1:nz, j in 1:na
        cash = r*assets[j] + income[j, z]
        best = -Inf
        chosen = max(cash, 1e-10)
        for side in (-1, 1)
            k = j + side
            1 <= k <= na || continue
            slope = (V[k, z] - V[j, z])/(assets[k] - assets[j])
            candidate = max(slope, 1e-10)^(-1/γ)
            d = cash - candidate
            if (side == 1 && d >= 0) || (side == -1 && d <= 0)
                value = utility(candidate, γ) + d*slope
                if value > best
                    best, chosen = value, candidate
                end
            end
        end
        if cash > 0 && utility(cash, γ) > best
            chosen = cash
        end
        if j == 1
            chosen = min(chosen, max(cash, 1e-10))
        elseif j == na
            chosen = max(chosen, cash)
        end
        c[j, z] = chosen
        drift[j, z] = cash - chosen
    end
    return c, drift
end

function solve_age_hjb(next, assets, ds, age, r, income, mp, Q;
        retired = false, retirement_value = nothing, max_policy_iter = 40,
        policy_tol = 1e-8, backend = :cpu)
    na, nz = size(next)
    V = copy(next)
    mortal = mp.m(age)
    bequests = bequest.(assets, Ref(mp))
    for _ in 1:max_policy_iter
        c, drift = consumption_policy(V, assets, r, income, mp.γ)
        G = retired ? asset_generator(drift, assets) : worker_generator(drift, assets, Q)
        flow = [utility(c[j, z], mp.γ) + mortal*bequests[j] -
                (retired ? 0.0 : mp.ψ(age)) for j in 1:na, z in 1:nz]
        rhs = vec(flow) .+ vec(next)./ds
        M = (1/ds + mp.ρ + mortal)*sparse(I, na*nz, na*nz) - G
        if retirement_value !== nothing
            outside = vec(repeat(retirement_value, 1, nz))
            # Howard selection between the continuation equation and obstacle.
            active = vec(V) .- outside .<= ds.*(M*vec(V) .- rhs)
            M = spdiagm(0 => Float64.(.!active))*M +
                spdiagm(0 => Float64.(active))
            rhs = ifelse.(active, outside, rhs)
        end
        candidate = reshape(sparse_system_solve(M, rhs, backend), na, nz)
        gap = maximum(abs, candidate .- V)
        V = candidate
        gap < policy_tol && return V
    end
    throw(ErrorException("HJB policy iteration failed at age $age; refine grid or increase max_policy_iter"))
end

function backward_policies(mp, ages, assets, pensions,
        r, wage, transfer, tax_rate, Q;
        max_policy_iter = 80, policy_tol = 1e-8, backend = :cpu)
    retired_assets = pensions.retired_assets
    na, nr, nz, ns = length(assets), length(retired_assets), length(mp.z_vals), length(ages)
    np = length(pensions.monthly_benefits)
    VL = zeros(nr, ns)
    VM = fill(NaN, nr, np, ns)
    VW = zeros(na, nz, ns)
    retirement_type = zeros(UInt8, na, nz, ns) # 0 work, 1 lump, 2 monthly, 3 no benefit
    terminal = bequest.(retired_assets, Ref(mp))
    VL[:, end] .= terminal
    VM[:, :, end] .= terminal
    lump_income = fill(transfer, nr, 1)
    monthly_income = repeat(reshape(transfer .+ pensions.monthly_benefits, 1, np), nr, 1)
    for i in ns:-1:1
        if i < ns
            ds = ages[i+1] - ages[i]
            VL[:, i] .= vec(solve_age_hjb(reshape(VL[:, i+1], nr, 1), retired_assets,
                ds, ages[i], r, lump_income, mp, Q; retired = true,
                max_policy_iter, policy_tol, backend))
            # Only states whose entry age has been reached are needed here.
            active = findall(s -> s <= ages[i], pensions.monthly_first_age)
            VM[:, active, i] .= solve_age_hjb(VM[:, active, i+1], retired_assets,
                ds, ages[i], r, monthly_income[:, active], mp, Q; retired = true,
                max_policy_iter, policy_tol, backend)
        end
        lump_option = pensions.lump_shifts[i]*VL[:, i]
        monthly_option = VM[1:na, pensions.monthly_state[i], i]
        outside = max.(lump_option, monthly_option)
        payment_type = ifelse.(lump_option .> monthly_option, UInt8(1), UInt8(2))
        if !(mp.Tr_lb <= ages[i] <= mp.Tr_ub)
            outside = VL[1:na, i]
            payment_type .= UInt8(3)
        end
        if i == ns
            VW[:, :, i] .= outside
            retirement_type[:, :, i] .= payment_type
            continue
        end
        ds = ages[i+1] - ages[i]
        earnings = wage*exp(mp.z_bar(ages[i])) .* exp.(mp.z_vals)
        net_income = earnings .- pension_tax.(Ref(mp), tax_rate, earnings) .+ transfer
        worker_income = repeat(reshape(net_income, 1, nz), na, 1)
        obstacle = ages[i] >= mp.Tw ? reshape(outside, na, 1) : nothing
        VW[:, :, i] .= solve_age_hjb(VW[:, :, i+1], assets, ds,
            ages[i], r, worker_income, mp, Q; retirement_value = obstacle,
            max_policy_iter, policy_tol, backend)
        if ages[i] >= mp.Tw
            for z in 1:nz, j in 1:na
                if VW[j, z, i] <= outside[j] + 1e-9
                    retirement_type[j, z, i] = payment_type[j]
                end
            end
        end
    end
    return (; VL, VM, VW, retirement_type)
end

function forward_distribution(mp, ages, assets, pensions,
        population, values, r, wage, transfer, tax_rate, Q, pi; backend = :cpu)
    retired_assets = pensions.retired_assets
    na, nr, nz, ns = length(assets), length(retired_assets), length(mp.z_vals), length(ages)
    np = length(pensions.monthly_benefits)
    workers = zeros(na, nz, ns)
    lump_retirees = zeros(nr, ns)
    nonbenefit_retirees = zeros(nr, ns)
    monthly_by_benefit = zeros(nr, np, ns)
    lump_entries = zeros(ns)
    monthly_entries = zeros(ns)
    nonbenefit_entries = zeros(ns)
    start = findfirst(s -> s >= mp.Tw, ages)
    workers[1, :, start] .= pi
    K = L = bequest_flow = tax_base = monthly_benefits = lump_benefits = 0.0
    for i in start:ns
        age = ages[i]
        wmass = @view workers[:, :, i]
        lmass = @view lump_retirees[:, i]
        nmass = @view nonbenefit_retirees[:, i]
        mmass = @view monthly_by_benefit[:, :, i]
        if age >= mp.Tw
            entering_lump = zeros(na)
            entering_monthly = zeros(na)
            for z in 1:nz, j in 1:na
                action = values.retirement_type[j, z, i]
                if action == 1
                    entering_lump[j] += wmass[j, z]
                    wmass[j, z] = 0.0
                elseif action == 2
                    entering_monthly[j] += wmass[j, z]
                    wmass[j, z] = 0.0
                elseif action == 3
                    nmass[j] += wmass[j, z]
                    nonbenefit_entries[i] += wmass[j, z]
                    wmass[j, z] = 0.0
                end
            end
            lmass .+= transpose(pensions.lump_shifts[i])*entering_lump
            mmass[1:na, pensions.monthly_state[i]] .+= entering_monthly
            lump_entries[i] = sum(entering_lump)
            monthly_entries[i] = sum(entering_monthly)
            # Events are discrete age masses, not age densities: no ds weight.
            lump_benefits += population.density[i]*pensions.lump_benefits[i]*lump_entries[i]
        end
        mass_weight = population.weights[i]*population.density[i]
        mean_assets = dot(assets, vec(sum(wmass; dims = 2))) +
            dot(retired_assets, lmass .+ nmass .+ vec(sum(mmass; dims = 2)))
        earnings = wage*exp(mp.z_bar(age)) .* exp.(mp.z_vals)
        worker_by_z = vec(sum(wmass; dims = 1))
        K += mass_weight*mean_assets
        bequest_flow += mass_weight*population.mortality[i]*mean_assets
        L += mass_weight*exp(mp.z_bar(age))*dot(exp.(mp.z_vals), worker_by_z)
        tax_base += mass_weight*dot(min.(earnings, mp.tax_cap), worker_by_z)
        monthly_benefits += mass_weight*dot(pensions.monthly_benefits, vec(sum(mmass; dims = 1)))
        i == ns && break
        ds = ages[i+1] - age
        _, drift = consumption_policy(reshape(values.VL[:, i], nr, 1), retired_assets,
            r, fill(transfer, nr, 1), mp.γ)
        G = asset_generator(drift, retired_assets)
        # No-benefit retirees and lump recipients share post-payment policies.
        propagated = sparse_system_solve(sparse(I, nr, nr) - ds*transpose(G),
            hcat(lmass, nmass), backend)
        lump_retirees[:, i+1] .= propagated[:, 1]
        nonbenefit_retirees[:, i+1] .= propagated[:, 2]
        active = findall(p -> sum(mmass[:, p]) > 0, 1:np)
        if !isempty(active)
            income = repeat(reshape(transfer .+ pensions.monthly_benefits[active], 1, :), nr, 1)
            _, md = consumption_policy(values.VM[:, active, i], retired_assets, r, income, mp.γ)
            GM = asset_generator(md, retired_assets)
            monthly_by_benefit[:, active, i+1] .= reshape(sparse_system_solve(
                sparse(I, nr*length(active), nr*length(active)) - ds*transpose(GM),
                vec(mmass[:, active]), backend), nr, length(active))
        end
        if age < mp.T
            net_income = earnings .- pension_tax.(Ref(mp), tax_rate, earnings) .+ transfer
            income = repeat(reshape(net_income, 1, nz), na, 1)
            _, wd = consumption_policy(values.VW[:, :, i], assets, r, income, mp.γ)
            GW = worker_generator(wd, assets, Q)
            workers[:, :, i+1] .= reshape(sparse_system_solve(
                sparse(I, na*nz, na*nz) - ds*transpose(GW), vec(wmass), backend), na, nz)
        end
    end
    monthly_retirees = dropdims(sum(monthly_by_benefit; dims = 2); dims = 2)
    terminal_assets = dot(assets, vec(sum(workers[:, :, end]; dims = 2))) +
        dot(retired_assets, lump_retirees[:, end] .+ nonbenefit_retirees[:, end] .+ monthly_retirees[:, end])
    bequest_flow += population.density[end]*terminal_assets
    upper_asset_mass = sum(population.weights .* population.density .*
        [sum(workers[end, :, i]) + lump_retirees[end, i] + nonbenefit_retirees[end, i] + monthly_retirees[end, i]
         for i in eachindex(ages)])
    tax_revenue = tax_rate*tax_base
    pension_outlays = lump_benefits + monthly_benefits
    return (; K, L, bequest_flow, upper_asset_mass, workers,
        lump_retirees, nonbenefit_retirees, monthly_retirees, monthly_by_benefit,
        lump_entries, monthly_entries, nonbenefit_entries,
        tax_base, tax_revenue, lump_benefits, monthly_benefits, pension_outlays)
end

# The equilibrium has three unknowns. A damped Broyden step avoids the
# dozens of full lifecycle solves used by pseudo-time integration.
function broyden_equilibrium(f, u0; tol, maxiters = 50)
    u = copy(u0)
    F = f(u)
    all(isfinite, F) || error("Nonfinite initial equilibrium residual")
    n = length(u)
    J = zeros(n, n)
    function refresh_jacobian!()
        for j in 1:n
            shifted = copy(u)
            h = 1e-3*max(1.0, abs(u[j]))
            shifted[j] += h
            J[:, j] .= (f(shifted) .- F)./h
        end
    end
    refresh_jacobian!()
    for iteration in 1:maxiters
        maximum(abs, F) <= tol &&
            return (; u, retcode = :Success, iterations = iteration - 1)
        accepted = false
        for attempt in 1:2
            attempt == 2 && refresh_jacobian!()
            step = try
                -(J \ F)
            catch
                fill(NaN, n)
            end
            all(isfinite, step) || continue
            step ./= max(1.0, maximum(abs, step)/0.75)
            for backtrack in 0:8
                δ = (2.0^-backtrack).*step
                candidate_u = u .+ δ
                candidate_F = f(candidate_u)
                if all(isfinite, candidate_F) &&
                        maximum(abs, candidate_F) < maximum(abs, F)
                    J .+= ((candidate_F .- F .- J*δ)*transpose(δ))./dot(δ, δ)
                    u, F = candidate_u, candidate_F
                    accepted = true
                    break
                end
            end
            accepted && break
        end
        accepted || return (; u, retcode = :Failure, iterations = iteration)
    end
    return (; u, retcode = :Failure, iterations = maxiters)
end

"""
    solve_stationary_equilibrium(mp; n_age=201, retirement_age_refinement=5,
        n_assets=240, a_max=100.0, tax_rate_guess=0.6)

Solve a stationary zero-reserve pension system. An endogenous capped payroll
rate balances lump-sum retirement payments and continuing monthly pensions.
Households choose their retirement age and payment type irreversibly.
Capital, bequest transfers, and the pension budget are solved jointly with
Broyden, with `DynamicSS(Tsit5())` as fallback. `backend=:gpu` uses CUDA sparse
QR; the present sparse systems are faster on the default CPU backend.

`n_age` sets the base age grid. Its intervals inside the retirement window
are subdivided by `retirement_age_refinement`. The asset grid concentrates
`dense_grid_share` of intervals below `dense_asset_fraction*a_max`.
Retirees get an extra asset-grid tail of the largest eligible lump sum so the jump
is represented without clipping. The lump-sum interpolation and its transpose
are used for values and mass, respectively.
"""
function solve_stationary_equilibrium(mp; n_age = 201,
        retirement_age_refinement = 5, n_assets = 240, a_max = 100.0,
        dense_asset_fraction = 0.7, dense_grid_share = 0.9,
        capital_labor_guess = 45.0, transfer_guess = 0.25, tax_rate_guess = 0.6,
        abstol = 1e-6, reltol = 1e-6, max_policy_iter = 80,
        policy_tol = 1e-8, backend = :cpu, equilibrium_method = :auto,
        verbose = false)
    equilibrium_method in (:auto, :broyden, :dynamic) ||
        throw(ArgumentError("equilibrium_method must be :auto, :broyden or :dynamic"))
    backend in (:cpu, :gpu) || throw(ArgumentError("backend must be :cpu or :gpu"))
    n_age >= 3 && n_assets >= 3 || throw(ArgumentError("grids need at least 3 points"))
    a_max > 0 && capital_labor_guess > 0 && transfer_guess > 0 ||
        throw(ArgumentError("asset bound and initial guesses must be positive"))
    mp.γ > 0 && mp.ρ > 0 && mp.κ > 0 || throw(ArgumentError("γ, ρ, κ must be positive"))
    isfinite(mp.p) && isfinite(mp.l) && mp.p >= 0 && mp.l >= 0 &&
        mp.tax_cap > 0 && 0 <= mp.τ0 < 1 ||
        throw(ArgumentError("benefits must be nonnegative, tax_cap positive, and τ0 in [0,1)"))
    validate_retirement_settings(mp)
    pension_active = mp.p > 0 || mp.l > 0
    !pension_active || 0 < tax_rate_guess < 1 ||
        throw(ArgumentError("tax_rate_guess must be between 0 and 1"))
    ages = age_grid(mp, n_age, retirement_age_refinement)
    assets = asset_grid(n_assets, a_max, dense_asset_fraction, dense_grid_share)
    pensions = pension_states(mp, ages, assets)
    retired_assets = pensions.retired_assets
    Q = productivity_generator(mp)
    pi = invariant_distribution(Q)
    population = age_population(mp, ages)
    evaluations = Ref(0)
    function evaluate(u)
        evaluations[] += 1
        kl, transfer = exp.(u[1:2])
        tax_rate = pension_active ? 1/(1 + exp(-u[3])) : 0.0
        r = mp.α*mp.A*kl^(mp.α-1)
        wage = (1-mp.α)*mp.A*kl^mp.α
        values = backward_policies(mp, ages, assets, pensions,
            r, wage, transfer, tax_rate, Q; max_policy_iter, policy_tol, backend)
        dist = forward_distribution(mp, ages, assets, pensions,
            population, values, r, wage, transfer, tax_rate, Q, pi; backend)
        if verbose
            println((evaluation = evaluations[], capital_labor = kl,
                transfer, tax_rate, budget = dist.tax_revenue - dist.pension_outlays))
        end
        return (; kl, transfer, tax_rate, r, wage, values, dist)
    end
    function equilibrium_drift!(du, u, p, t)
        state = evaluate(u)
        du[1] = log(max(state.dist.K/state.dist.L, 1e-12)) - u[1]
        du[2] = log(max(state.dist.bequest_flow, 1e-12)) - u[2]
        if pension_active
            du[3] = log(max(state.dist.pension_outlays, 1e-12)/
                max(state.dist.tax_revenue, 1e-12))
        end
    end
    initial = log.([capital_labor_guess, transfer_guess])
    pension_active && push!(initial, log(tax_rate_guess/(1-tax_rate_guess)))
    function residual_at(u)
        F = similar(u)
        equilibrium_drift!(F, u, nothing, 0.0)
        return F
    end
    sol = equilibrium_method === :dynamic ? nothing :
        broyden_equilibrium(residual_at, initial; tol = max(abstol, reltol))
    if sol === nothing || (sol.retcode !== :Success && equilibrium_method === :auto)
        start = sol === nothing ? initial : sol.u
        problem = SteadyStateProblem(equilibrium_drift!, start)
        sol = solve(problem, DynamicSS(Tsit5()); abstol, reltol)
    end
    string(sol.retcode) == "Success" ||
        error("stationary equilibrium did not converge: $(sol.retcode)")
    state = evaluate(sol.u)
    dense_asset_cutoff = dense_asset_fraction*a_max
    last_dense = searchsortedlast(assets, dense_asset_cutoff)
    last_retired_dense = searchsortedlast(retired_assets, dense_asset_cutoff)
    retirees = state.dist.lump_retirees .+ state.dist.nonbenefit_retirees .+ state.dist.monthly_retirees
    age_weights = population.weights .* population.density
    adult_mass = sum(age_weights[i]*(sum(state.dist.workers[:, :, i]) +
        sum(retirees[:, i])) for i in eachindex(ages))
    tail_mass = sum(age_weights[i]*(sum(state.dist.workers[last_dense+1:end, :, i]) +
        sum(retirees[last_retired_dense+1:end, i])) for i in eachindex(ages))
    asset_mass_above_dense = tail_mass/adult_mass
    if asset_mass_above_dense > 0.05
        @warn "More than 5% of adults hold assets above the dense grid; increase dense_asset_fraction" asset_mass_above_dense
    end
    if state.dist.upper_asset_mass > 1e-3
        @warn "Asset distribution reaches the upper boundary; increase a_max or n_assets" upper_asset_mass=state.dist.upper_asset_mass
    end
    residual = [log(state.dist.K/state.dist.L/state.kl),
        log(state.dist.bequest_flow/state.transfer)]
    pension_active && push!(residual, log(state.dist.pension_outlays/state.dist.tax_revenue))
    pension_budget_residual = state.dist.tax_revenue - state.dist.pension_outlays
    return (; ages, assets, retired_assets, lump_shift = pensions.lump_shifts,
        monthly_benefits = pensions.monthly_benefits,
        monthly_state = pensions.monthly_state,
        monthly_first_age = pensions.monthly_first_age,
        lump_benefits_by_age = pensions.lump_benefits,
        population, growth = population.η, interest = state.r, wage = state.wage,
        transfer = state.transfer, tax_rate = state.tax_rate, τ0 = state.tax_rate,
        pension_tax = y -> pension_tax(mp, state.tax_rate, y),
        capital = state.dist.K, labor = state.dist.L, capital_labor = state.kl,
        bequest_flow = state.dist.bequest_flow, pension_budget_residual,
        pension_revenue = state.dist.tax_revenue,
        pension_outlays = state.dist.pension_outlays,
        lump_sum_outlays = state.dist.lump_benefits,
        monthly_outlays = state.dist.monthly_benefits,
        upper_asset_mass = state.dist.upper_asset_mass,
        dense_asset_cutoff, dense_grid_points = last_dense, asset_mass_above_dense,
        value_lump = state.values.VL, value_monthly = state.values.VM,
        value_retired = hcat([mp.Tr_lb <= ages[i] <= mp.Tr_ub ?
            max.(pensions.lump_shifts[i]*state.values.VL[:, i],
                state.values.VM[1:length(assets), pensions.monthly_state[i], i]) :
            state.values.VL[1:length(assets), i] for i in eachindex(ages)]...),
        value_working = state.values.VW, workers = state.dist.workers, retirees,
        lump_retirees = state.dist.lump_retirees,
        nonbenefit_retirees = state.dist.nonbenefit_retirees,
        monthly_retirees = state.dist.monthly_retirees,
        monthly_retirees_by_benefit = state.dist.monthly_by_benefit,
        nonbenefit_retirement_entries = state.dist.nonbenefit_entries,
        lump_retirement_entries = state.dist.lump_entries,
        monthly_retirement_entries = state.dist.monthly_entries,
        retirement_type = state.values.retirement_type,
        retirement = state.values.retirement_type .> 0,
        residual, solution = sol, backend, equilibrium_evaluations = evaluations[])
end

# Policies are evaluated in their post-payment asset states. Lump-sum
# retirement curves use the entry-age lump sum against pre-retirement assets.
function stationary_policies(res, mp)
    na, nr, nz, ns = length(res.assets), length(res.retired_assets),
        length(mp.z_vals), length(res.ages)
    worker_consumption = fill(NaN, na, nz, ns)
    worker_saving = fill(NaN, na, nz, ns)
    lump_consumption = fill(NaN, nr, ns)
    lump_saving = fill(NaN, nr, ns)
    monthly_consumption = fill(NaN, size(res.value_monthly))
    monthly_saving = fill(NaN, size(res.value_monthly))
    for i in eachindex(res.ages)
        age = res.ages[i]
        c, drift = consumption_policy(reshape(res.value_lump[:, i], nr, 1),
            res.retired_assets, res.interest, fill(res.transfer, nr, 1), mp.γ)
        lump_consumption[:, i] .= vec(c)
        lump_saving[:, i] .= vec(drift)
        active = findall(s -> s <= age, res.monthly_first_age)
        income = repeat(reshape(res.transfer .+ res.monthly_benefits[active], 1, :), nr, 1)
        c, drift = consumption_policy(res.value_monthly[:, active, i],
            res.retired_assets, res.interest, income, mp.γ)
        monthly_consumption[:, active, i] .= c
        monthly_saving[:, active, i] .= drift
        if mp.Tw <= age < mp.T
            earnings = res.wage*exp(mp.z_bar(age)) .* exp.(mp.z_vals)
            net_income = earnings .- res.pension_tax.(earnings) .+ res.transfer
            income = repeat(reshape(net_income, 1, nz), na, 1)
            c, drift = consumption_policy(res.value_working[:, :, i], res.assets,
                res.interest, income, mp.γ)
            for z in 1:nz, j in 1:na
                if res.retirement_type[j, z, i] == 0
                    worker_consumption[j, z, i] = c[j, z]
                    worker_saving[j, z, i] = drift[j, z]
                end
            end
        end
    end
    return (; worker_consumption, worker_saving, lump_consumption, lump_saving,
        monthly_consumption, monthly_saving)
end

productivity_label(z) = isinf(z) ? "Unemployed" : "z = $(round(z; digits = 3))"

function chosen_age_curve(res, worker_values, lump_values, monthly_values, z, age_index)
    action = res.retirement_type[:, z, age_index]
    lump = res.lump_shift[age_index]*lump_values[:, age_index]
    monthly = monthly_values[1:length(res.assets), res.monthly_state[age_index], age_index]
    return ifelse.(action .== 1, lump,
        ifelse.(action .== 2, monthly,
            ifelse.(action .== 3, lump_values[1:length(res.assets), age_index],
                worker_values[:, z, age_index])))
end

function age_slice_figure(res, mp, worker_values, lump_values, monthly_values;
        title, ylabel, target_ages = (35.0, 50.0, 61.0, 80.0))
    age_indices = unique([argmin(abs.(res.ages .- target)) for target in target_ages
        if first(res.ages) <= target <= last(res.ages)])
    filter!(i -> res.ages[i] >= mp.Tw, age_indices)
    isempty(age_indices) && throw(ArgumentError("No plot ages fall in the model"))
    nrows = cld(length(age_indices), 2)
    fig = Figure(size = (1550, 460*nrows))
    worker_colors = (:steelblue, :seagreen, :purple, :goldenrod, :crimson, :teal)
    first_worker_axis = nothing
    for (panel, age_index) in enumerate(age_indices)
        row, col = cld(panel, 2), mod1(panel, 2)
        age = res.ages[age_index]
        status = age < mp.T ? " (retirement choice)" : " (terminal age)"
        ax = Axis(fig[row, col], title = "Age $(round(age; digits = 1))$status",
            xlabel = "Assets (million NTD)", ylabel = ylabel)
        if age < mp.T
            first_worker_axis === nothing && (first_worker_axis = ax)
            for z in eachindex(mp.z_vals)
                curve = chosen_age_curve(res, worker_values, lump_values,
                    monthly_values, z, age_index)
                lines!(ax, res.assets, curve;
                    color = worker_colors[mod1(z, length(worker_colors))],
                    linewidth = 2.5, label = productivity_label(mp.z_vals[z]))
            end
        end
        if age > mp.Tr_ub
            lines!(ax, res.retired_assets, lump_values[:, age_index];
                color = :darkorange, linewidth = 2.5, label = "Lump-sum / no-benefit retirees")
            for (origin, color) in zip((mp.Tr_lb, mp.Tr, mp.Tr_ub), (:seagreen, :teal, :purple))
                entry_index = argmin(abs.(res.ages .- origin))
                pindex = res.monthly_state[entry_index]
                lines!(ax, res.retired_assets, monthly_values[:, pindex, age_index];
                    color, linewidth = 2.5, linestyle = :dash,
                    label = "Monthly: retired at $(res.ages[entry_index])")
            end
            axislegend(ax; position = :rb)
        end
    end
    first_worker_axis === nothing || Legend(fig[1:nrows, 3],
        first_worker_axis, "Productivity state"; framevisible = false)
    Label(fig[0, 1:2], title, fontsize = 24)
    return fig
end

function retirement_plot_data(res, mp)
    indices = findall(s -> mp.Tw <= s < mp.T, res.ages)
    lump_mass = vec(sum(res.lump_retirees; dims = 1))[indices]
    monthly_mass = vec(sum(res.monthly_retirees; dims = 1))[indices]
    nonbenefit_mass = vec(sum(res.nonbenefit_retirees; dims = 1))[indices]
    worker_mass = [sum(res.workers[:, :, i]) for i in indices]
    lump_flow = res.lump_retirement_entries[indices] .* res.population.density[indices]
    monthly_flow = res.monthly_retirement_entries[indices] .* res.population.density[indices]
    nonbenefit_flow = res.nonbenefit_retirement_entries[indices] .* res.population.density[indices]
    total_flow = sum(lump_flow .+ monthly_flow .+ nonbenefit_flow)
    total_flow > 0 || error("No retirement events before terminal age")
    total_mass = lump_mass .+ monthly_mass .+ nonbenefit_mass .+ worker_mass
    return (; ages = res.ages[indices],
        lump_probability = lump_flow ./ total_flow,
        monthly_probability = monthly_flow ./ total_flow,
        nonbenefit_probability = nonbenefit_flow ./ total_flow,
        retirement_age_probability = (lump_flow .+ monthly_flow .+ nonbenefit_flow) ./ total_flow,
        lump_percentage = 100 .* lump_mass ./ total_mass,
        monthly_percentage = 100 .* monthly_mass ./ total_mass,
        nonbenefit_percentage = 100 .* nonbenefit_mass ./ total_mass,
        retired_percentage = 100 .* (lump_mass .+ monthly_mass .+ nonbenefit_mass) ./ total_mass)
end

# Cross-sectional wage observations are indexed by age and productivity;
# asset masses are summed because wages do not depend on assets within a cell.
function wage_distribution_data(res, mp; n_bins = 50)
    n_bins >= 1 || throw(ArgumentError("n_bins must be positive"))
    wages = Float64[]
    masses = Float64[]
    unemployed_mass = 0.0
    age_weights = res.population.weights .* res.population.density
    for i in eachindex(res.ages), z in eachindex(mp.z_vals)
        mass = age_weights[i]*sum(res.workers[:, z, i])
        mass > 0 || continue
        if mp.z_vals[z] == -Inf
            unemployed_mass += mass
        end
        push!(wages, res.wage*exp(mp.z_bar(res.ages[i]) + mp.z_vals[z]))
        push!(masses, mass)
    end
    total_mass = sum(masses)
    total_mass > 0 || error("No workers in the stationary distribution")
    positive_mass = sum(masses[j] for j in eachindex(wages) if wages[j] > 0)
    positive_mass > 0 || error("No positive wages in the stationary distribution")
    max_wage = maximum(wages)
    edges = collect(range(0.0, max_wage, length = n_bins + 1))
    bin_width = edges[2] - edges[1]
    bin_mass = zeros(n_bins)
    for j in eachindex(wages)
        wages[j] > 0 || continue
        bin = min(searchsortedlast(edges, wages[j]), n_bins)
        bin_mass[bin] += masses[j]
    end
    bin_centers = (edges[1:end-1] .+ edges[2:end])./2
    positive_density = bin_mass ./ (positive_mass*bin_width)
    unemployment_rate = 100*unemployed_mass/total_mass
    return (; bin_centers, bin_width, positive_density, unemployment_rate)
end

"""
    plot_stationary_equilibrium(res, mp; output_dir=joinpath(@__DIR__, "..", "..", "figure", "continued_pension"))

Save value, consumption, saving, retirement, stationary-distribution, and
asset-group comparison SVGs. Each value or policy panel shows one of ages
35, 50, 61, and 80, with productivity-state curves overlaid where applicable.
Each curve selects the working or retired outcome at every asset
according to the equilibrium retirement decision. At age 80, additional
retiree curves compare pensions fixed at retirement ages 60, 65, and 70. Further SVGs show the
retirement-age distribution, retired percentage by age, and wage distribution.
"""
function plot_stationary_equilibrium(res, mp;
        output_dir = normpath(joinpath(@__DIR__, "..", "..", "figure", "continued_pension")))
    mkpath(output_dir)
    paths = (
        values = joinpath(output_dir, "extension_value_functions.svg"),
        consumption = joinpath(output_dir, "extension_consumption_policy.svg"),
        saving = joinpath(output_dir, "extension_saving_policy.svg"),
        retirement = joinpath(output_dir, "extension_retirement_region.svg"),
        retirement_age_distribution = joinpath(output_dir,
            "extension_retirement_age_distribution.svg"),
        retired_percentage = joinpath(output_dir,
            "extension_retired_percentage_by_age.svg"),
        distribution = joinpath(output_dir, "extension_stationary_distribution.svg"),
        asset_groups = joinpath(output_dir, "extension_asset_distributions.svg"),
        wages = joinpath(output_dir, "extension_wage_distribution.svg"),
    )
    na, nz, ns = length(res.assets), length(mp.z_vals), length(res.ages)
    value_fig = age_slice_figure(res, mp, res.value_working, res.value_lump, res.value_monthly;
        title = "Chosen value by age and productivity", ylabel = "Value")
    save(paths.values, value_fig)

    policies = stationary_policies(res, mp)
    consumption_fig = age_slice_figure(res, mp,
        policies.worker_consumption, policies.lump_consumption, policies.monthly_consumption;
        title = "Chosen consumption by age and productivity",
        ylabel = "Consumption (million NTD/year)")
    save(paths.consumption, consumption_fig)

    saving_fig = age_slice_figure(res, mp,
        policies.worker_saving, policies.lump_saving, policies.monthly_saving;
        title = "Chosen saving by age and productivity",
        ylabel = "Saving (million NTD/year)")
    save(paths.saving, saving_fig)

    # Retirement decisions are shown over all adult ages, including ineligible ages.
    retirement_ages = findall(s -> mp.Tw <= s < mp.T, res.ages)
    retirement_fig = Figure(size = (1500, 750))
    decision_plot = nothing
    for z in 1:nz
        row, col = cld(z, 3), mod1(z, 3)
        ax = Axis(retirement_fig[row, col],
            title = productivity_label(mp.z_vals[z]),
            xlabel = "Age", ylabel = "Assets (million NTD)")
        xlims!(ax, 60, 70)
        decision_plot = heatmap!(ax, res.ages[retirement_ages], res.assets,
            permutedims(Float64.(res.retirement_type[:, z, retirement_ages]));
            colormap = cgrad([:steelblue, :darkorange, :seagreen, :gray]; categorical = true),
            colorrange = (-0.5, 3.5), rasterize = 2)
    end
    Colorbar(retirement_fig[1:2, 4], decision_plot,
        ticks = ([0.0, 1.0, 2.0, 3.0],
            ["Continue working", "Lump-sum", "Monthly pension", "No pension"]))
    Label(retirement_fig[0, 1:3], "Retirement decision by productivity state",
        fontsize = 24)
    save(paths.retirement, retirement_fig)

    retirement_stats = retirement_plot_data(res, mp)
    retirement_age_fig = Figure(size = (1000, 520))
    age_ax = Axis(retirement_age_fig[1, 1],
        title = "Distribution of retirement ages",
        xlabel = "Retirement age", ylabel = "Share of retirement events (%)")
    bar_width = length(retirement_stats.ages) > 1 ?
        0.85*minimum(diff(retirement_stats.ages)) : 0.5
    barplot!(age_ax, retirement_stats.ages .- bar_width/4,
        100 .* retirement_stats.lump_probability;
        width = bar_width/2, color = :darkorange, label = "Lump-sum")
    barplot!(age_ax, retirement_stats.ages .+ bar_width/4,
        100 .* retirement_stats.monthly_probability;
        width = bar_width/2, color = :seagreen, label = "Monthly pension")
    barplot!(age_ax, retirement_stats.ages, 100 .* retirement_stats.nonbenefit_probability;
        width = bar_width/2, color = :gray, label = "No pension")
    axislegend(age_ax; position = :rt)
    xlims!(age_ax, 60, 70)
    save(paths.retirement_age_distribution, retirement_age_fig)

    retired_share_fig = Figure(size = (1000, 520))
    share_ax = Axis(retired_share_fig[1, 1],
        title = "Retired share by age",
        xlabel = "Age", ylabel = "Retired among living people at age (%)")
    lines!(share_ax, retirement_stats.ages, retirement_stats.lump_percentage;
        color = :darkorange, linewidth = 2.5, label = "Lump-sum retirees")
    lines!(share_ax, retirement_stats.ages, retirement_stats.monthly_percentage;
        color = :seagreen, linewidth = 2.5, label = "Monthly pensioners")
    lines!(share_ax, retirement_stats.ages, retirement_stats.nonbenefit_percentage;
        color = :gray, linewidth = 2.5, label = "No-pension retirees")
    lines!(share_ax, retirement_stats.ages, retirement_stats.retired_percentage;
        color = :black, linewidth = 2.5, linestyle = :dash, label = "All retirees")
    axislegend(share_ax; position = :rb)
    xlims!(share_ax, 60, 70)
    ylims!(share_ax, 0, 100)
    save(paths.retired_percentage, retired_share_fig)

    wage_stats = wage_distribution_data(res, mp)
    wage_fig = Figure(size = (1000, 560))
    wage_density_ax = Axis(wage_fig[1, 1],
        title = "Wages among employed workers",
        xlabel = "Wage (million NTD/year)",
        ylabel = "Density among positive-wage workers")
    barplot!(wage_density_ax, wage_stats.bin_centers,
        wage_stats.positive_density;
        width = wage_stats.bin_width, color = :steelblue)
    Label(wage_fig[0, 1],
        "Stationary wage distribution (unemployment rate: $(round(wage_stats.unemployment_rate; digits = 1))%)",
        fontsize = 24)
    save(paths.wages, wage_fig)

    # Compare population densities on the two asset grids; each node mass is
    # divided by its own cell width. The total uses the extended retiree grid.
    nr = length(res.retired_assets)
    worker_mass = dropdims(sum(res.workers; dims = 2); dims = 2)
    age_density = reshape(res.population.density, 1, ns)
    worker_density = worker_mass .* age_density ./
        reshape(asset_cell_widths(res.assets), na, 1)
    retired_scale = age_density ./ reshape(asset_cell_widths(res.retired_assets), nr, 1)
    lump_density = res.lump_retirees .* retired_scale
    monthly_density = res.monthly_retirees .* retired_scale
    nonbenefit_density = res.nonbenefit_retirees .* retired_scale
    total_mass = copy(res.retirees)
    total_mass[1:na, :] .+= worker_mass
    total_density = total_mass .* retired_scale
    distribution_fig = Figure(size = (2350, 490))
    density_plot = nothing
    for (col, (label, density, grid)) in enumerate((
            ("Working", worker_density, res.assets),
            ("Lump-sum retirees", lump_density, res.retired_assets),
            ("Monthly pensioners", monthly_density, res.retired_assets),
            ("No-pension retirees", nonbenefit_density, res.retired_assets),
            ("Total", total_density, res.retired_assets)))
        ax = Axis(distribution_fig[1, col], title = label,
            xlabel = "Age", ylabel = "Assets (million NTD)")
        density_plot = heatmap!(ax, res.ages, grid,
            permutedims(log10.(max.(density, 1e-10)));
            colormap = :viridis, colorrange = (-10.0, log10(maximum(total_density))),
            rasterize = 2)
    end
    Colorbar(distribution_fig[1, 6], density_plot,
        label = "log10 population density / year / million NTD")
    Label(distribution_fig[0, 1:5], "Stationary age and asset distribution",
        fontsize = 24)
    save(paths.distribution, distribution_fig)

    age_weights = res.population.weights .* res.population.density
    worker_asset_mass = reshape(reshape(res.workers, na*nz, ns)*age_weights, na, nz)
    lump_asset_mass = res.lump_retirees*age_weights
    monthly_asset_mass = res.monthly_retirees*age_weights
    nonbenefit_asset_mass = res.nonbenefit_retirees*age_weights
    overall_asset_mass = lump_asset_mass .+ monthly_asset_mass .+ nonbenefit_asset_mass
    overall_asset_mass[1:na] .+= vec(sum(worker_asset_mass; dims = 2))
    asset_fig = Figure(size = (1700, 650))
    density_ax = Axis(asset_fig[1, 1], title = "Asset density",
        xlabel = "Assets (million NTD)", ylabel = "Conditional density / million NTD")
    ylims!(density_ax, 0, 0.2)
    cdf_ax = Axis(asset_fig[1, 2], title = "Cumulative asset distribution",
        xlabel = "Assets (million NTD)",
        ylabel = "Share of group with assets at or below level")
    ylims!(cdf_ax, 0, 1.02)
    function add_asset_group!(mass, grid, label, color; linewidth = 2, linestyle = :solid)
        group_mass = sum(mass)
        group_mass > 0 || return
        probability = mass ./ group_mass
        lines!(density_ax, grid, probability ./ asset_cell_widths(grid);
            label, color, linewidth, linestyle)
        lines!(cdf_ax, grid, cumsum(probability); color, linewidth, linestyle)
    end
    worker_colors = (:steelblue, :seagreen, :purple, :goldenrod, :crimson, :teal)
    for z in 1:nz
        add_asset_group!(view(worker_asset_mass, :, z), res.assets,
            "Working: $(productivity_label(mp.z_vals[z]))",
            worker_colors[mod1(z, length(worker_colors))])
    end
    add_asset_group!(lump_asset_mass, res.retired_assets, "Lump-sum retirees",
        :darkorange; linewidth = 3, linestyle = :dash)
    add_asset_group!(monthly_asset_mass, res.retired_assets, "Monthly pensioners",
        :seagreen; linewidth = 3, linestyle = :dot)
    add_asset_group!(nonbenefit_asset_mass, res.retired_assets, "No-pension retirees",
        :gray; linewidth = 3, linestyle = :dashdot)
    add_asset_group!(overall_asset_mass, res.retired_assets, "Overall", :black; linewidth = 3)
    Legend(asset_fig[1, 3], density_ax, "Groups"; framevisible = false)
    Label(asset_fig[0, 1:3], "Stationary asset distributions by group", fontsize = 24)
    save(paths.asset_groups, asset_fig)
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    mp = model()
    res = solve_stationary_equilibrium(mp)
    figure_paths = plot_stationary_equilibrium(res, mp)
end
