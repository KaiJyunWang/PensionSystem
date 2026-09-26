# Shared retirement-age benefit schedule from paper/extension.tex.
# p and l are the annual pension and lump sum at the standard retirement age Tr.
function pension_multiplier(mp, retirement_age)
    mp.Tr_lb <= retirement_age <= mp.Tr_ub || return 0.0
    return 1 + mp.benefit_adjustment*(retirement_age - mp.Tr)
end

function pension_benefits(mp, retirement_age)
    factor = pension_multiplier(mp, retirement_age)
    return (; monthly = factor*mp.p, lump = factor*mp.l)
end

# Each attainable monthly amount is an absorbing state: aging never changes it.
function pension_states(mp, ages, assets)
    benefits = pension_benefits.(Ref(mp), ages)
    monthly_benefits = sort(unique([benefit.monthly for benefit in benefits]))
    monthly_state = [findfirst(==(benefit.monthly), monthly_benefits) for benefit in benefits]
    monthly_first_age = [minimum(ages[monthly_state .== p]) for p in eachindex(monthly_benefits)]
    lump_benefits = [benefit.lump for benefit in benefits]
    retired_assets = retirement_asset_grid(assets, maximum(lump_benefits))
    lump_shifts = [lump_sum_operator(assets, retired_assets, lump) for lump in lump_benefits]
    return (; monthly_benefits, monthly_state, monthly_first_age, lump_benefits, retired_assets, lump_shifts)
end

function validate_retirement_settings(mp)
    0 <= mp.Tw <= mp.Tr_lb <= mp.Tr <= mp.Tr_ub < mp.T ||
        throw(ArgumentError("require 0 ≤ Tw ≤ Tr_lb ≤ Tr ≤ Tr_ub < T"))
    isfinite(mp.benefit_adjustment) &&
        min(pension_multiplier(mp, mp.Tr_lb), pension_multiplier(mp, mp.Tr_ub)) >= 0 ||
        throw(ArgumentError("retirement benefit multipliers must be finite and nonnegative"))
    return nothing
end
