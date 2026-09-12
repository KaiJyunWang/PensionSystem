module ParameterTables

using Printf

export write_parameter_table

"""
    write_parameter_table(para; output_path = "./table/parameters.tex", panel = :both)

Write a complete LaTeX parameter table. Select `panel = :A` for external
calibration, `:B` for internal calibration, or `:both` (the default).
Internal targets describe the moments matched by `loss` in
`stochastic_pension_run.jl`; both preference parameters match the choice-share
time series jointly. Only fields belonging to the selected panel are required.

```julia
include("parameter_tables.jl")
using .ParameterTables
write_parameter_table(para; panel = :A, output_path = "table/parameters_A.tex")
write_parameter_table(para; panel = :B, output_path = "table/parameters_B.tex")
```
"""
function write_parameter_table(para; output_path = "./table/parameters.tex", panel = :both)
    panel in (:A, :B, :both) || throw(ArgumentError("panel must be :A, :B, or :both"))

    external_calibration = panel == :B ? [] : [
        (raw"$b$",    "Birth rate",                para.b,  "Total fertility rate"),
        (raw"$m$",    "Mortality rate",            para.m,  "Median lifespan"),
        (raw"$T_w$",  "Working age",               para.Tw, ""),
        (raw"$T_r$",  "Retirement age",            para.T,  "Standard retirement age"),
        (raw"$T_m$",  "Age of mortality exposure", para.Tm, ""),
        (raw"$\rho$", "Subjective discount rate",  para.ρ,  "Interest rate"),
        (raw"$\tau$", "Pension tax",               para.τ,  "Pension tax"),
        (raw"$y$",    "Income",                    para.y,  "Highest monthly insured salary"),
        (raw"$\ell$", "Lump-sum pension",          para.l,  "Transfer based on 30 years of working"),
        (raw"$p$",    "Monthly pension",           para.p,  "Pension based on 30 years of working"),
    ]
    internal_calibration = panel == :A ? [] : [
        (raw"$r$",       "Pension fund return rate",       para.r, raw"$\E\sbrc{\frac{\hat{B}_{t+\Delta}-\hat{B}_t}{B_t}-\mu_{B}(\hat{B}_t, \hat{Q}_t, \theta)\Delta} = 0$"),
        (raw"$\sigma$", "Pension fund volatility",        para.σ, raw"$\text{SD}(\frac{\hat{B}_{t+\Delta}-\hat{B}_t}{B_t\sqrt{\Delta}}) = \sigma$"),
        (raw"$\alpha$", "Preference for lump-sum scheme", para.α, raw"$\E\sbrc{(q - q(\hat{B}_t, \hat{Q}_t, \theta))^2}$"),
        (raw"$\beta$",  "Sensitivity to default risk",    para.β, raw"$\E\sbrc{(q - q(\hat{B}_t, \hat{Q}_t, \theta))^2}$"),
    ]

    value_string(x) = x isa Integer ? string(x) : @sprintf("%.4g", x)
    linebreak = repeat("\\", 2)

    function write_panel(io, title, rows)
        println(io, "        \\multicolumn{4}{l}{\\textit{", title, "}} ", linebreak)
        println(io, raw"        \addlinespace")
        for (symbol, description, value, target) in rows
            println(io, "        ", symbol, " & ", description, " & \$",
                value_string(value), "\$ & ", target, " ", linebreak)
        end
    end

    open(output_path, "w") do io
        println(io, raw"\begin{threeparttable}")
        println(io, raw"    \begin{tabular}{clcl}")
        println(io, raw"        \toprule")
        println(io, "        Parameter & Description & Value & Target ", linebreak)
        println(io, raw"        \midrule")
        if panel in (:A, :both)
            write_panel(io, "Panel A: External Calibration", external_calibration)
        end
        if panel == :both
            println(io, raw"        \addlinespace")
            println(io, raw"        \midrule")
        end
        if panel in (:B, :both)
            write_panel(io, "Panel B: Internal Calibration", internal_calibration)
        end
        println(io, raw"        \bottomrule")
        println(io, raw"    \end{tabular}")
        println(io)
        println(io, raw"    \begin{tablenotes}[flushleft]")
        println(io, raw"        \footnotesize")
        println(io, raw"        \item \textit{Note:} The monetary unit in the model is 100,000 NTD.")
        println(io, raw"    \end{tablenotes}")
        println(io, raw"\end{threeparttable}")
    end

    return output_path
end

end # module ParameterTables
