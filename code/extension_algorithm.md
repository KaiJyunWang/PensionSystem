# Stationary equilibrium with a balanced pension budget

Two standalone scripts preserve the two pension scenarios. Run them in separate Julia processes from the repository root:

    julia code/extension_collapsed.jl
    julia code/extension_continued.jl

| Script | Pension assumption | Default figure directory |
| --- | --- | --- |
| `code/extension_collapsed.jl` | After the fund reaches zero, taxes and both pension benefits stop permanently. | `figure/collapsed_pension/` |
| `code/extension_continued.jl` | Current contributions fund lump-sum and monthly benefits; the tax rate balances the budget. | `figure/continued_pension/` |

`code/extension.jl` is a compatibility entry point that includes the continued pension script, so `julia code/extension.jl` still runs that scenario. Both standalone scripts construct `mp = model()`, solve the stationary equilibrium, and write nine SVG figures. Their function and global names are shared, so load each scenario in a separate Julia session. For another parameter set:

    res = solve_stationary_equilibrium(mp)
    paths = plot_stationary_equilibrium(res, mp)

The pension system continues to operate with zero reserves. Current payroll contributions finance current lump-sum retirement payments and ongoing monthly pensions. The benefits `mp.l` and `mp.p` remain in force, while the payroll rate is determined in equilibrium. The supplied `mp.τ0` is the baseline rate; `res.tax_rate` (also `res.τ0`) is the solved rate, and `res.pension_tax(y)` evaluates the equilibrium tax. The solver does not mutate `mp`.

## Preserved collapsed pension scenario

`code/extension_collapsed.jl` restores the implementation saved before introducing the balanced pension budget. It retains the refined age grid, concentrated asset grid, CPU/GPU backends, and all nine original plots. Its household budgets use gross labor earnings plus bequest transfers for workers, and bequest transfers alone for retirees; `mp.τ`, `mp.p`, and `mp.l` are not applied in the collapsed equilibrium. There is one retired state and no payment-type choice or lump-sum asset jump.

The collapsed solver solves two residuals, `log((K/L)/k)` and `log(bequest_flow/tr)`, for capital per effective worker and bequest transfers. Its retirement plots show total retirement decisions and shares; value and policy figures have one retired curve. The remaining sections describe the continued pension implementation.

## States, prices, and taxes

Workers have age $s$, nonnegative assets $a$, and productivity state $z$. After retiring they remain in one of two absorbing payment states:

- Lump-sum recipients receive $\ell$ once at retirement and no subsequent pension income.
- Monthly pensioners receive the annual flow $p$ while alive.

The off-diagonal entries of `mp.Λ` give productivity jump rates. The code reconstructs the diagonal so the generator $Q$ has zero row sums. Its invariant distribution, named `pi`, is the productivity distribution at working age $T_w$. The state $z=-\infty$ represents unemployment.

For a trial capital-to-effective-labor ratio $k$, the firm conditions give

$$
r=\alpha A k^{\alpha-1},\qquad w=(1-\alpha)Ak^\alpha.
$$

Gross earnings are $y(s,z)=w\exp(\bar z(s)+z)$. The pension contribution is

$$
\tau(y)=\tau_0\min(y,\bar y),
$$

where `tax_cap` is $\bar y$ and defaults to 0.550 million NTD per year. Working disposable income is $y-\tau(y)+tr$. The equilibrium tax rate is parameterized by a logistic transformation, keeping $0<\tau_0<1$. If the calibration cannot balance its pension budget in this interval, the solver must fail rather than report an equilibrium with a negative worker cash flow induced by a tax exceeding earnings.

## Population and age grid

The stationary population density is

$$
\bar n(s)=b\exp\left[-\eta s-\int_0^s m(x)\,dx\right].
$$

Mortality is integrated with the trapezoid rule. Population growth $\eta$ is found by bisection so the age density integrates to one. Age integrals use nonuniform trapezoid weights $\omega_i$.

The base age grid has `n_age=201` equally spaced nodes from 0 to $T$. The default `retirement_age_refinement=5` splits every base interval in the retirement window into five steps. At the default ages 60–70 this gives 0.1-year steps in the window, 0.5-year steps elsewhere, and 281 nodes in total. `Tw`, `Tr_lb`, and `Tr_ub` must be on the base grid. Both backward and forward equations use the actual interval $\Delta s_i$. Set the refinement to 1 to recover the base grid.

## Asset grids and the lump-sum jump

The worker asset grid has 240 nodes by default. Approximately `dense_grid_share=0.9` of its intervals cover $[0,\text{dense_asset_fraction}\,a_{\max}]$, where the default fraction is 0.7 and $a_{\max}=100$. The remaining intervals cover the upper tail. Neighboring grid distances are used in every derivative and generator transition.

The retiree grid contains all worker nodes and extends to $a_{\max}+\ell$. This permits the asset jump $a\mapsto a+\ell$ even at the largest worker node. It prevents clipping a pension payment at the artificial grid boundary. The retired value and mass arrays therefore have slightly more asset rows than the worker arrays; use `res.retired_assets` for them.

Let $S$ linearly interpolate retiree-grid values at $a_j+\ell$. Each row of $S$ sums to one, and

$$
S\,a^r=a^w+\ell.
$$

The lump-sum retirement option is $SV^\ell$. Entering lump-sum mass is moved with $S^\mathsf T$. Using the same operator for values and its adjoint for mass preserves both probability and the asset payment. Monthly recipients enter at their existing asset node.

The returned grid diagnostics include `dense_asset_cutoff`, `dense_grid_points`, `asset_mass_above_dense`, and `upper_asset_mass`. Warnings flag material mass beyond the dense region or at either asset boundary. Adjust `a_max`, grid concentration, or node counts when changing the calibration.

## Household values and retirement choices

For post-payment lump-sum recipients and monthly pensioners, respectively,

$$
d^\ell=ra+tr-c^\ell,\qquad d^m=ra+tr+p-c^m.
$$

Both retired value equations have the form

$$
(\rho+m(s))V^R=\max_c\{u(c)+m(s)b_q(a)+\partial_sV^R+d^R\partial_aV^R\},
$$

where $R\in\{\ell,m\}$. Terminal values at maximum age $T$ equal bequest utility. The working continuation equation uses

$$
d^w=ra+y-\tau(y)+tr-c^w
$$

and adds the working disutility $-\psi(s)$ and productivity term $(QV^w)_z$.

Before `Tr_lb`, work is mandatory, as in the previous solver. Within the retirement window, the household can continue working or choose the larger of

$$
V^\ell(a+\ell,s),\qquad V^m(a,s).
$$

At `Tr_ub`, retirement is mandatory, but the payment choice remains optimal. Payment types cannot be switched after retirement. Exact ties between payment types select monthly payments.

Age is marched backward. Asset derivatives use first-order upwind differences according to the proposed asset drift. The utility first-order condition gives $c=(V_a)^{-1/\gamma}$ on each admissible side; the code compares those choices with zero drift. At zero assets the borrowing constraint is enforced; at the upper boundary drift is constrained inward.

For fixed consumption policy, an age step solves

$$
\left[(\Delta s_i^{-1}+\rho+m(s_i))I-G_i\right]V_i
=f_i+\Delta s_i^{-1}V_{i+1}.
$$

A Howard active-set iteration chooses between continuation rows and the retirement obstacle. Retirement rows impose the best retirement value; other rows impose the HJB equation. Consumption and the active set are updated until the value change is below `policy_tol`. `max_policy_iter` defaults to 80.

`res.retirement_type` stores 0 for work, 1 for lump-sum retirement, and 2 for monthly retirement. `res.retirement` is the corresponding Boolean mask. The main value fields are `value_working`, `value_lump`, and `value_monthly`; `value_retired` is the best retirement-entry option on the worker asset grid, retained for compatibility.

## Forward distributions and aggregates

At $T_w$, workers start with zero assets and productivity probabilities `pi`. At each eligible age, the decision map moves new retirees into their chosen payment state. The solver records their conditional cohort masses in `lump_retirement_entries` and `monthly_retirement_entries`.

Each remaining distribution advances through the transpose of its conservative generator:

$$
\mu_{i+1}=(I-\Delta s_iG_i^\mathsf T)^{-1}\mu_i.
$$

Monthly and lump-sum retirees have separate generators because their income flows differ. Mortality and population growth enter the stationary age weights, so the conditional distributions themselves do not lose mass to mortality. Total worker and retiree conditional mass remains one at every modeled adult age.

`workers` is on `res.assets`; `lump_retirees` and `monthly_retirees` are on `res.retired_assets`. `retirees` is their sum. Capital aggregates household assets across all three states, while effective labor aggregates only workers. With zero pension reserves, household assets equal productive capital. Bequest transfers equal assets left at age-specific deaths, including the terminal deaths at $T$; the transfer is distributed to the entire normalized living population.

## Pension budget and equilibrium iteration

For a trial distribution, the contribution base and monthly outlays are

$$
D=\sum_i\omega_i\bar n(s_i)\sum_{a,z}\min(y(s_i,z),\bar y)\,\mu^w_{a,z,i},
\qquad
P_m=p\sum_i\omega_i\bar n(s_i)\sum_a\mu^m_{a,i}.
$$

If $e_i^\ell$ is the conditional mass newly retiring with a lump sum at node $i$, then

$$
P_\ell=\ell\sum_i\bar n(s_i)e_i^\ell.
$$

Lump-sum entries are discrete retirement events, so their budget contribution uses population density at the event age without an extra trapezoid age weight. Multiplying them by $\omega_i$ would incorrectly shrink payments as the retirement grid is refined. Continuing monthly payments are a stock of recipients and use the age integration weights.

The balanced-budget condition is

$$
\tau_0D=P_\ell+P_m.
$$

The three unknowns are $\log k$, $\log tr$, and the logit of $\tau_0$. Every trial solves household values and payment choices backward, distributions forward, then targets

$$
\log\left(\frac{\hat K/\hat L}{k}\right)=0,\quad
\log\left(\frac{\text{bequest flow}}{tr}\right)=0,\quad
\log\left(\frac{P_\ell+P_m}{\tau_0D}\right)=0.
$$

A damped Broyden method uses a finite-difference Jacobian, bounded steps, and backtracking. Automatic mode falls back to SteadyStateDiffEq's `DynamicSS(Tsit5())` if Broyden fails; `equilibrium_method=:broyden` or `:dynamic` selects either method explicitly. With both pension benefits zero, the rate is zero and the budget unknown is dropped.

Results report `tax_rate`, `pension_revenue`, `pension_outlays`, `lump_sum_outlays`, `monthly_outlays`, `pension_budget_residual`, `residual`, and `equilibrium_evaluations`. `verbose=true` prints trial quantities and budget gaps. Inspect residuals and compare grids when numerical accuracy matters.

## CPU and GPU execution

`backend=:cpu` uses Julia's sparse direct solve. `backend=:gpu` lazily loads CUDA.jl, converts each system to CUDA CSR format, and uses cuSOLVER sparse QR before copying the result back. CUDA.jl and a working CUDA device are required. On the earlier 1,440-state calibration, CPU sparse solves were faster than GPU sparse QR, so CPU remains the default.

Lifecycle steps are sequential in age with one coupled worker system per age. DiffEqGPU's independent-trajectory ensemble interface does not directly accelerate this structure. See the [DiffEqGPU ensemble documentation](https://docs.sciml.ai/DiffEqGPU/stable/manual/ensemblegpuarray/) and [CUDA sparse linear algebra documentation](https://cuda.juliagpu.org/stable/lib/cusparse/).

## SVG output

- extension_value_functions.svg overlays productivity curves at ages 35, 50, and 61. The age-61 curve selects work, lump-sum retirement, or monthly retirement. At age 80, separate curves show existing lump-sum recipients and monthly pensioners.
- extension_consumption_policy.svg uses the same layout for chosen consumption. At retirement, the lump-sum curve is evaluated after the asset jump $a+\ell$.
- extension_saving_policy.svg shows the corresponding continuous asset drift. The one-time lump-sum jump is separate from this drift.
- extension_retirement_region.svg shows work, lump-sum retirement, and monthly retirement across age and assets for each productivity state.
- extension_retirement_age_distribution.svg shows stationary new-retirement flows by age, separately for lump-sum and monthly choices. Their probabilities jointly sum to one.
- extension_retired_percentage_by_age.svg shows lump-sum, monthly, and total retired shares conditional on each age from 60 to 70, reaching 100% total at mandatory retirement.
- extension_stationary_distribution.svg shows worker, lump-sum, monthly, and overall age–asset densities. Each node mass is divided by its asset-cell width; the color scale is logarithmic with a $10^{-10}$ floor.
- extension_asset_distributions.svg compares workers by productivity, lump-sum recipients, monthly pensioners, and the overall adult population. Age-integrated groups are normalized individually. Density is displayed over 0–0.2, and the second panel shows cumulative shares. Groups with zero population mass are omitted from these conditional curves.
- extension_wage_distribution.svg shows the positive-wage density among employed workers. The title reports unemployment as stationary mass in $z=-\infty$ divided by all nonretired worker mass. Retirees are excluded.

At an eligible age, the selected outcome is $X^w$ for type 0, $X^\ell(a+\ell)$ for type 1, and $X^m(a)$ for type 2. The interpolation operator is applied to lump-sum values and policies; the code selects the active policy directly to avoid `0*NaN`. At age 80, asset axes describe current post-payment holdings. The nearest available grid point is used for each requested plotting age.

Heatmaps are raster layers inside SVG files. Lines, bars, axes, and labels remain vector graphics. Monetary axes use millions of NTD; ages and flows use years.
