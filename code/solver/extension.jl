# Compatibility entry point for the continued, balanced-budget pension model.
include(joinpath(@__DIR__, "extension_continued.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    mp = model()
    res = solve_stationary_equilibrium(mp)
    figure_paths = plot_stationary_equilibrium(res, mp)
end
