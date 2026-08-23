using LinearAlgebra
using SparseArrays

function get_model_parameters(;
    # Preference parameters defaults
    γ = 2.0,                     # CRRA risk aversion coefficient
    ρ = 0.04,                    # Subjective discount rate
    
    # Economic / Environment parameters defaults
    r = 0.03,                    # Real interest rate
    p = 1.5,                     # Retirement / Pension income
    
    # Bequest motive parameters defaults
    θ = 2.0,                     # Bequest motive strength
    κ = 0.5,                     # Bequest luxury parameter (κ > 0)
    
    # Gompertz-Makeham Mortality parameters defaults
    m1 = 0.0005,
    m2 = 0.09,
    m3 = 0.001,

    # Horizon parameters defaults
    age_start = 65.0,
    age_end = 95.0,
    
    # Numerical grid parameters defaults
    Nt = 600,                    # Time steps
    Na = 150,                    # Asset steps
    amax = 50.0                  # Upper bound of asset grid
)
    # Pack everything up and return it cleanly as a NamedTuple
    return (; γ, ρ, r, p, θ, κ, m1, m2, m3, age_start, age_end, Nt, Na, amax)
end

