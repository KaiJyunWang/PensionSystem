# Run from the repository root: julia code/solver/extension_checks.jl
using Test, LinearAlgebra

module CollapsedExtension
include(joinpath(@__DIR__, "extension_collapsed.jl"))
end
module ContinuedExtension
include(joinpath(@__DIR__, "extension_continued.jl"))
end

const CE = ContinuedExtension
const CO = CollapsedExtension

function check_equilibrium(res, mp)
    @test maximum(abs, res.residual) < 1e-6
    @test isapprox(dot(res.population.weights, res.population.density), 1; atol = 1e-12)
    @test isapprox(res.capital/res.labor, res.capital_labor; rtol = 1e-6)
    @test isapprox(res.bequest_flow, res.transfer; rtol = 1e-6)
    start = findfirst(>=(mp.Tw), res.ages)
    for i in start:length(res.ages)
        @test isapprox(sum(res.workers[:, :, i]) + sum(res.retirees[:, i]), 1; atol = 1e-10)
    end
    @test minimum(res.workers) >= -1e-12
    @test minimum(res.retirees) >= -1e-12
    if hasproperty(res, :monthly_retirees_by_benefit)
        weights = res.population.weights .* res.population.density
        # Recompute payments from the public cohort distributions and schedule.
        monthly = sum(weights[i]*sum(res.monthly_benefits[p]*
            sum(res.monthly_retirees_by_benefit[:, p, i])
            for p in eachindex(res.monthly_benefits)) for i in eachindex(res.ages))
        lump = sum(res.population.density[i]*CE.pension_benefits(mp, res.ages[i]).lump*
            res.lump_retirement_entries[i] for i in eachindex(res.ages))
        @test isapprox(monthly, res.monthly_outlays; atol = 1e-12)
        @test isapprox(lump, res.lump_sum_outlays; atol = 1e-12)
        @test abs(res.pension_budget_residual) < 1e-6
        @test isapprox(res.pension_revenue, monthly + lump; atol = 1e-6)
        @test res.monthly_retirees ≈ dropdims(sum(res.monthly_retirees_by_benefit; dims = 2); dims = 2)
        for i in eachindex(res.ages)
            if !(mp.Tr_lb <= res.ages[i] <= mp.Tr_ub)
                @test res.lump_retirement_entries[i] == 0
                @test res.monthly_retirement_entries[i] == 0
            end
        end
    else
        @test res.tax_rate == res.pension_outlays == res.pension_revenue == 0
    end
end

@testset "Retirement-age schedule and asset payments" begin
    mp = CE.model()
    for (age, factor) in ((60, 0.8), (65, 1.0), (70, 1.2))
        benefit = CE.pension_benefits(mp, age)
        @test benefit.monthly ≈ factor*mp.p
        @test benefit.lump ≈ factor*mp.l
    end
    for age in (20, 59.9, 70.1, 80)
        @test CE.pension_benefits(mp, age) == (monthly = 0.0, lump = 0.0)
    end
    @test CE.model(Tr = 66).Tr_lb == 61
    @test_throws ArgumentError CE.validate_retirement_settings(CE.model(benefit_adjustment = 0.3))
    assets = CE.asset_grid(40, 100.0, 0.7, 0.9)
    for refinement in (1, 5)
        ages = CE.age_grid(mp, 101, refinement)
        pensions = CE.pension_states(mp, ages, assets)
        @test last(pensions.retired_assets) ≈ 100 + 1.2*mp.l
        for i in eachindex(ages)
            S = pensions.lump_shifts[i]
            @test vec(sum(S; dims = 2)) ≈ ones(length(assets))
            @test S*pensions.retired_assets ≈ assets .+ CE.pension_benefits(mp, ages[i]).lump
        end
    end
end

@testset "Locked pensions and permanent ineligibility" begin
    mp = CE.model()
    ages = CE.age_grid(mp, 101, 1)
    assets = CE.asset_grid(40, 100.0, 0.7, 0.9)
    pensions = CE.pension_states(mp, ages, assets)
    population = CE.age_population(mp, ages)
    Q = CE.productivity_generator(mp)
    pi = CE.invariant_distribution(Q)
    r, wage, transfer, tax = 0.025, 2.4, 0.3, 0.45
    values = CE.backward_policies(mp, ages, assets, pensions, r, wage, transfer, tax, Q)
    for (retire_age, action) in ((60, 2), (70, 2), (60, 1), (70, 1), (59, 3), (71, 3))
        entry = findfirst(==(retire_age), ages)
        values.retirement_type .= 0
        values.retirement_type[:, :, entry] .= action
        dist = CE.forward_distribution(mp, ages, assets, pensions, population,
            values, r, wage, transfer, tax, Q, pi)
        if action == 2
            pindex = pensions.monthly_state[entry]
            @test sum(dist.monthly_entries) ≈ 1
            @test sum(dist.monthly_by_benefit[:, pindex, end]) ≈ 1
            @test sum(dist.monthly_by_benefit[:, setdiff(1:length(pensions.monthly_benefits), [pindex]), :]) == 0
            expected = CE.pension_benefits(mp, retire_age).monthly*
                sum(population.weights[entry:end] .* population.density[entry:end])
            @test dist.monthly_benefits ≈ expected
            @test sum(dist.monthly_by_benefit[:, pindex, findfirst(==(80), ages)]) ≈ 1
        elseif action == 1
            @test dist.lump_benefits ≈ population.density[entry]*CE.pension_benefits(mp, retire_age).lump
            @test dist.monthly_benefits == 0
            @test sum(dist.lump_retirees[:, end]) ≈ 1
        else
            @test dist.pension_outlays == 0
            @test sum(dist.lump_entries) == sum(dist.monthly_entries) == 0
            @test dist.nonbenefit_entries[entry] ≈ 1
            @test sum(dist.nonbenefit_retirees[:, end]) ≈ 1
        end
    end
end

@testset "Stationary equilibria and zero-benefit limit" begin
    grids = (; n_age = 101, retirement_age_refinement = 1, n_assets = 80,
        equilibrium_method = :broyden)
    collapsed = CO.solve_stationary_equilibrium(CO.model(); grids...)
    continued = CE.solve_stationary_equilibrium(CE.model(); grids...)
    check_equilibrium(collapsed, CO.model())
    check_equilibrium(continued, CE.model())
    zero = CE.solve_stationary_equilibrium(CE.model(p = 0.0, l = 0.0); grids...)
    check_equilibrium(zero, CE.model(p = 0.0, l = 0.0))
    @test zero.tax_rate == 0
    @test zero.capital ≈ collapsed.capital rtol = 1e-6
    @test zero.labor ≈ collapsed.labor rtol = 1e-6
    @test zero.transfer ≈ collapsed.transfer rtol = 1e-6
    @test zero.value_working ≈ collapsed.value_working rtol = 1e-6
end
