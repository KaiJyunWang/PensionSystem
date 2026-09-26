# Stationary equilibrium with retirement-age-dependent benefits

The implementation follows `paper/extension.tex`. Run the scenarios separately:

```sh
julia code/solver/extension_collapsed.jl
julia code/solver/extension_continued.jl
```

`code/solver/extension.jl` runs the continued scenario for compatibility. Each entry point solves its equilibrium and writes nine SVGs under `figure/collapsed_pension/` or `figure/continued_pension/`. Including a solver defines functions without starting a solve or writing figures. Both scenarios share function names, so use separate Julia sessions or modules.

```julia
include("code/solver/extension_continued.jl")
mp = model()
res = solve_stationary_equilibrium(mp)
paths = plot_stationary_equilibrium(res, mp)
```

## Confirmed model settings

The standard retirement age is `Tr=65`; the benefit window defaults to `Tr_lb=Tr-5` and `Tr_ub=Tr+5`, including both endpoints. The baseline annual pension `p=0.256` and lump sum `l=2.061` are in million NTD and apply to retirement at 65. The common annual adjustment is `benefit_adjustment=0.04`:

$$
f(q)=1+0.04(q-T_r),\qquad p(q)=f(q)\bar p,\qquad \ell(q)=f(q)\bar\ell.
$$

For retirement ages 60, 65, and 70, the multipliers are 0.8, 1.0, and 1.2. `pension_benefits(mp, q)` returns the corresponding `monthly` and `lump` amounts. Both amounts are zero outside the window.

Workers may retire irreversibly at **any adult age** from `Tw=20` until `T=100`. There is no forced retirement at 70. Retiring outside the window gives up all future pension claims: aging into the window does not confer eligibility. Within the window, households choose between a lump sum and a monthly pension. The monthly amount is fixed for the rest of life at its retirement-age level; it does not increase as the retiree ages. Exact payment-type ties select monthly payments.

In the collapsed scenario taxes and all pension payments are permanently zero, including for existing retirees. There is a single retired state. The same unrestricted retirement rule applies.

The continued scenario has zero pension reserves, so current contributions finance current payments. Its endogenous tax rate balances the pension budget. `mp.τ0=0.12` is the baseline input; `res.tax_rate` (also `res.τ0`) is the equilibrium rate. The code leaves `mp` unchanged. The existing calibration, bequest preferences, working disutility, terminal bequest condition, production function, and tax cap remain in use.

## State grids and population

Workers have assets, age, and productivity. Lump-sum recipients and retirees without benefits have the same post-payment value function but separate distributions. Monthly recipients additionally carry their permanently fixed pension amount as an absorbing state.

The base grid defaults to `n_age=201`. `Tw`, `Tr_lb`, and `Tr_ub` must be base-grid nodes. `retirement_age_refinement=5` subdivides the 60–70 window to 0.1-year spacing, leaving 0.5-year spacing elsewhere (281 age nodes). Retirement remains available outside the refined window. Custom `Tr` changes the default window automatically.

The stationary age density is

$$
\bar n(s)=b\exp\left[-\eta s-\int_0^s m(x)\,dx\right].
$$

Mortality is integrated by trapezoids; bisection finds population growth $\eta$ so the density integrates to one. Nonuniform trapezoid weights $\omega_i$ are used in age integrals. Conditional adult distributions have unit mass; mortality and population growth enter through age weights rather than through conditional asset/productivity transitions. Residual survivors at `T` leave terminal bequests, as in the existing finite-horizon implementation. Terminal liquidation is excluded from the retirement-age plots.

The asset grid defaults to 240 nodes and `a_max=100`. `dense_grid_share=0.9` of the intervals lie below `dense_asset_fraction*a_max=70`. Every derivative uses its actual neighboring grid spacing. The retiree grid contains the worker grid and extends to `a_max + maximum(ℓ(q))`, which is 102.4732 by default.

For each entry age $q_i$, a distinct interpolation matrix $S_i$ maps retiree values at $a+\ell(q_i)$ to the worker grid. Its rows sum to one and $S_i a^r=a^w+\ell(q_i)$. New lump-sum mass moves with $S_i^\mathsf T$, preserving probability and the asset payment. Monthly and ineligible retirees enter at their existing asset node.

`extension_benefits.jl` provides the shared benefit schedule. Every attainable monthly amount on the retirement grid gets a state; no pension interpolation is needed. With the default positive age adjustment, there are 101 positive amounts and one zero amount. Monthly values before an amount's first possible entry are undefined (`NaN`) and are never used by the solver.

## Household solution

For a trial capital/labor ratio $k$, firm prices are

$$
r=\alpha A k^{\alpha-1},\qquad w=(1-\alpha)Ak^\alpha.
$$

Earnings are $y(s,z)=w\exp(\bar z(s)+z)$ and contributions are $\tau(y)=\tau_0\min(y,0.550)$. The off-diagonal entries of `Λ` specify productivity jumps; the diagonal is reconstructed for zero row sums. Its invariant distribution `pi` initializes productivity at `Tw`; `z=-Inf` represents unemployment. New workers have zero assets. Young people consume the uniform bequest transfer and hold no assets.

Retired asset drifts are $ra+tr-c$ for lump-sum and ineligible retirees and $ra+tr+p(q)-c$ for monthly retirees. The fixed $p(q)$ enters both the backward HJB and the corresponding forward generator. Working drift is $ra+y-\tau(y)+tr-c$; the HJB also includes productivity transitions and working disutility.

Within the window the retirement obstacle is

$$
\max\{V^\ell(a+\ell(q),s),\ V^m(a,s;p(q))\}.
$$

Outside the window it is $V^\ell(a,s)$ with no asset payment. A retiree can neither resume work nor switch payment types.

Age is marched backward with upwind asset differences. The utility condition gives $c=(V_a)^{-1/\gamma}$ on each admissible drift side; the code compares these candidates with zero drift. Borrowing is prohibited and drift is constrained inward at the artificial upper asset boundary.

For each policy, the implicit age step is

$$
[(\Delta s_i^{-1}+\rho+m(s_i))I-G_i]V_i
=f_i+\Delta s_i^{-1}V_{i+1}.
$$

Both scenarios use a Howard active-set solve for the retirement complementarity condition. Retirement rows impose the obstacle, and continuation rows impose the HJB. Policies and the active set iterate until `policy_tol=1e-8`. Monthly states are independent conditional on their fixed pension and are solved together as separate blocks.

## Forward solution and budget

At each adult age, the retirement map removes new retirees from the worker distribution and puts them in their absorbing state. Each remaining conditional distribution advances by

$$
\mu_{i+1}=(I-\Delta s_i G_i^\mathsf T)^{-1}\mu_i.
$$

With zero reserves, aggregate household assets equal productive capital. Labor includes workers only. The uniform transfer equals age-weighted assets left at deaths, including terminal deaths.

Pension revenue is $\tau_0D$ with

$$
D=\sum_i\omega_i\bar n(s_i)\sum_{a,z}\min(y(s_i,z),\bar y)\mu^w_{a,z,i}.
$$

For conditional lump-sum entries $e_i^\ell$ and monthly states $p_j$,

$$
P_\ell=\sum_i\bar n(s_i)\ell(s_i)e_i^\ell,
\qquad
P_m=\sum_i\omega_i\bar n(s_i)\sum_j p_j\sum_a\mu^m_{a,j,i}.
$$

Lump-sum entries are discrete retirement events and have **no additional age-step weight**. Monthly payments are a recipient stock and use trapezoid age weights. Monthly payments continue beyond age 70 at their original amount. Retirees without benefits contribute no pension expenditure.

The continued equilibrium solves jointly for $\log k$, $\log tr$, and the logit of $\tau_0$ using the residuals

$$
\log((K/L)/k),\qquad \log(\text{bequest flow}/tr),\qquad
\log((P_\ell+P_m)/(\tau_0D)).
$$

The logistic transformation restricts the rate to $(0,1)$; a calibration that cannot balance within this interval must fail. If both baseline benefits are zero, the tax is zero and the budget unknown is omitted. The collapsed economy solves only the first two residuals.

A damped Broyden method uses a finite-difference Jacobian and backtracking. `equilibrium_method=:auto` falls back to `DynamicSS(Tsit5())`; `:broyden` or `:dynamic` chooses explicitly. Every trial solves the lifecycle backward and forward. `verbose=true` prints continued-scenario trial prices and budget gaps. Grid diagnostics warn about material mass above the dense region or at the upper asset boundary.

## Results and figures

The continued result exposes:

| Field | Dimensions or meaning |
| --- | --- |
| `value_working`, `workers`, `retirement_type` | worker assets × productivity × age |
| `retirement_type` | 0 work, 1 lump sum, 2 monthly, 3 no benefits |
| `value_lump` | retiree assets × age; also the no-benefit value |
| `value_monthly` | retiree assets × fixed monthly amount × age |
| `monthly_benefits` | fixed annual amounts on the monthly-state axis |
| `monthly_state` | monthly-state index for retirement at each age node |
| `monthly_first_age` | earliest possible entry age for each monthly amount |
| `lump_shift` | vector of age-specific interpolation operators |
| `lump_benefits_by_age` | lump-sum amount for retirement at each age node |
| `value_retired` | best immediate retirement value on the worker grid |
| `lump_retirees`, `nonbenefit_retirees`, `monthly_retirees` | retiree assets × age |
| `monthly_retirees_by_benefit` | retiree assets × fixed monthly amount × age |
| `retirees` | sum of all three retired distributions |
| `lump_retirement_entries`, `monthly_retirement_entries`, `nonbenefit_retirement_entries` | conditional new-retirement masses by age |

`retirement` is true for any retired action. Prices, transfers, capital, labor, population growth, budget totals, residuals, and evaluation counts are also returned. The collapsed result has one `value_retired` and `retirees` array and explicitly reports zero tax and pension budget totals.

All nine SVGs are retained. Retirement maps, age distributions, and retired shares display ages 60–70; their underlying data still cover all adult ages. Continued figures distinguish retirement without benefits. Value/consumption/saving curves select the current action at ages 35, 50, 61, and 80; lump choices use the entry-age asset jump and monthly choices use the entry-age pension state. The age-80 panel additionally shows existing monthly retirees who entered at 60, 65, and 70. These cohorts may have different policies at the same current age and assets.

Distribution figures aggregate the monthly-benefit axis and display no-benefit retirees separately. Asset densities divide node masses by their asset-cell widths, and group CDFs normalize each nonempty group. Wage figures include working unemployment and exclude retirees. Heatmaps are raster layers within SVGs; other plot elements remain vectors.

`backend=:cpu` uses sparse direct solves. `backend=:gpu` retains the existing optional CUDA sparse QR path and requires CUDA.jl and a working GPU. CPU is the default; GPU execution has not been validated for the revised model.

## Validation and default numerical results

Run the reproducible numerical checks with:

```sh
julia code/solver/extension_checks.jl
```

They cover benefit-window endpoints, age-specific asset jumps on two retirement-grid refinements, fixed monthly cohorts through age 80, permanent ineligibility after retirement at 59 or 71, pension payment accounting, population mass conservation, market clearing, and equality with the collapsed economy when continued-system benefits are zero.

Both scenarios were also solved on the default 281-age-node, 240-worker-asset-node grid. The regenerated SVGs use these solutions. `table/extension_stationary_equilibria.csv` records prices, transfers, aggregates, budget totals, convergence residuals, and boundary mass for this calibration:

| Scenario | Interest rate | Contribution rate | Uniform annual transfer (million NTD) | Largest equilibrium residual |
| --- | ---: | ---: | ---: | ---: |
| Collapsed | 2.2387% | 0% | 0.334119 | 1.78e-7 |
| Continued | 2.6716% | 44.8905% | 0.258616 | 4.26e-8 |

The continued pension budget gap is -4.86e-10 million NTD per person per year. These are numerical equilibria for the stated grids, rather than a claim of convergence as all grid spacings go to zero. All reported monetary quantities use million NTD.
