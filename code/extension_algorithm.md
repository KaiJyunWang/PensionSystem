# Post-depletion stationary equilibrium solver

The implementation is in code/extension.jl. Run it from the repository root:

    julia code/extension.jl

The script constructs the default parameter set, solves the stationary equilibrium, and writes six SVG figures to figure/. The main entry point for another parameter set is:

    result = solve_stationary_equilibrium(mp)
    paths = plot_stationary_equilibrium(result, mp)

The solver assumes that after the per-person pension balance reaches zero, pension taxes, the lump-sum benefit, and monthly benefits remain zero. Those parameters remain in the parameter object for compatibility, but do not enter this stationary problem.

## State, prices, and population

A household's state is age $s$, nonnegative assets $a$, and status (working or retired). Workers also have productivity state $z$. The off-diagonal entries of the supplied $\Lambda$ are interpreted as jump intensities. The diagonal is reset to minus each row's outgoing rate, producing a conservative generator $Q$. The invariant distribution of $Q$, named pi in the code, is the productivity distribution at working age $T_w$.

For a trial capital-to-effective-labor ratio $k$ and bequest transfer $tr$, firm first-order conditions give

$$
r=\alpha A k^{\alpha-1},\qquad
w=(1-\alpha)A k^\alpha.
$$

The stationary age density per person is

$$
\bar n(s)=b\exp\left[-\eta s-\int_0^s m(x)\,dx\right].
$$

The solver integrates mortality on the age grid with the trapezoid rule. It then bisects on population growth $\eta$ until the discretized integral of $\bar n$ is one.

## Asset grid

The default asset grid is piecewise uniform. Its breakpoint is $a_*=f_a a_{\max}$, where $f_a$ is dense_asset_fraction. Of the $n_a-1$ intervals, approximately dense_grid_share of them cover $[0,a_*]$; the remaining intervals cover $(a_*,a_{\max}]$. The breakpoint and the upper endpoint are both included exactly. Every asset derivative and generator transition uses the actual neighboring grid distance, so the finite-difference equations remain valid on this nonuniform grid.

The choice follows a pilot stationary distribution on the original uniform grid: median assets were about 21.8 million NTD; the 99th and 99.99th percentiles were about 55.2 and 68.6 million NTD. Only about $5.3\times10^{-5}$ of adult mass lay above 70 million NTD. With $a_{\max}=100$, dense_asset_fraction $=0.7$, dense_grid_share $=0.9$, and 240 nodes, 216 nodes cover 0–70 with spacing about 0.326 million NTD. The remaining 24 nodes cover 70–100 with spacing 1.25 million NTD. This retains a tail and the asset boundary without assigning many nodes to scarcely occupied states.

With the concentrated grid, the default equilibrium puts about $2.5\times10^{-5}$ of adult mass above 70 million NTD and about $7.1\times10^{-13}$ of total population mass at the upper asset boundary. The returned result includes dense_asset_cutoff, dense_grid_points, and asset_mass_above_dense, the last measured as a fraction of adult population mass. A warning appears if more than 5% of adult mass lies above the breakpoint. The separate upper_asset_mass diagnostic warns when at least $10^{-3}$ of total population mass reaches the upper boundary. For a changed calibration, inspect both diagnostics and adjust dense_asset_fraction, dense_grid_share, or a_max. Compare equilibrium outputs across n_assets when numerical precision matters.

## Household values and policies

After depletion, a working household receives labor income $w\exp(\bar z(s)+z)$ and the common bequest transfer. A retired household receives the transfer. Their asset drifts are

$$
d^w=ra+w\exp(\bar z(s)+z)+tr-c^w,\qquad
d^r=ra+tr-c^r.
$$

Let $u(c)=c^{1-\gamma}/(1-\gamma)$, with the logarithmic limit when $\gamma=1$, and let $b_q(a)$ denote the bequest utility. The retired value satisfies

$$
(\rho+m(s))V^r
=\max_c\{u(c)+m(s)b_q(a)+\partial_s V^r+d^r\partial_a V^r\}.
$$

The working continuation value satisfies

$$
(\rho+m(s))V^w
=\max_c\{u(c)-\psi(s)+m(s)b_q(a)+\partial_s V^w
+d^w\partial_a V^w+(QV^w)_z\}.
$$

Before the lower retirement age, work is mandatory. Within the retirement window, the value is the larger of the working continuation value and $V^r$. At the upper retirement age, retirement is mandatory. At maximum age $T$, both values equal bequest utility because remaining households die there.

Age is marched backward. At age-grid point $s_i$, the age derivative is approximated by $(V_{i+1}-V_i)/\Delta s$. Asset derivatives use one-sided upwind differences: a positive asset drift uses the forward neighbor, and a negative drift uses the backward neighbor. The first-order condition proposes $c=(V_a)^{-1/\gamma}$ on each side. The code compares admissible candidates and the zero-drift choice. At $a=0$, consumption cannot exceed cash on hand; at the upper asset boundary, the drift is constrained inward.

For a fixed consumption and retirement policy, the sparse linear system is

$$
\left[\left(\Delta s^{-1}+\rho+m(s_i)\right)I-G_i\right]V_i
=f_i+\Delta s^{-1}V_{i+1},
$$

where $G_i$ combines upwind asset movements and, for workers, productivity jumps. Policy iteration updates consumption and the retirement obstacle until the value change is below the requested tolerance.

## Forward distribution and aggregates

At $T_w$, workers begin at zero assets with productivity probabilities pi. At each subsequent age, mass in cells choosing retirement is moved from the worker array to the retiree array. The remaining conditional asset distributions advance with an implicit step of the adjoint generator:

$$
\mu_{i+1}=(I-\Delta s\,G_i^\mathsf{T})^{-1}\mu_i.
$$

Age-specific mortality and population growth are accounted for by $\bar n(s_i)$ when aggregating, rather than by killing mass in the conditional distribution. Trapezoid weights over age produce capital and effective labor per person:

$$
\hat K=\int a\,d\mathfrak M,\qquad
\hat L=\int e^{\bar z(s)+z}\mathbf 1_{\{\mathrm{working}\}}\,d\mathfrak M.
$$

The bequest transfer equals the assets of households dying at each age, including those forced to die at $T$.

## Equilibrium iteration

The two equilibrium unknowns are $\log k$ and $\log tr$, keeping both positive. For each trial, the code solves household values backward, advances the distribution forward, and recomputes $\hat K/\hat L$ and the bequest flow. A damped two-variable Broyden method is the default. It initializes a finite-difference Jacobian, limits each log-price step, and backtracks until the residual falls. If it cannot make progress, the automatic mode falls back to SteadyStateDiffEq's DynamicSS(Tsit5()). The older dynamic method can also be selected explicitly with equilibrium_method=:dynamic. Both methods target these two log residuals:

$$
\log\left(\frac{\hat K/\hat L}{k}\right)=0,\qquad
\log\left(\frac{\text{bequest flow}}{tr}\right)=0.
$$

The closure equates household assets with productive capital after fund depletion.

The current defaults use 201 age points and the 240-point asset grid described above. The first-order upwind scheme can remain sensitive to asset spacing.

## CPU and GPU execution

The backend keyword accepts :cpu (default) or :gpu. The GPU path loads CUDA.jl only when requested, converts each sparse HJB or forward matrix to CUDA's CSR format, solves it with cuSOLVER sparse QR, and copies the solution back before the next age step. CUDA.jl and a working CUDA device are required for :gpu. CPU mode uses Julia's sparse direct solve.

The lifecycle recursions are sequential in age and involve one coupled worker system per age. DiffEqGPU.jl is designed for ensembles of independent differential equation trajectories, so it does not accelerate this structure directly. See the [DiffEqGPU ensemble documentation](https://docs.sciml.ai/DiffEqGPU/stable/manual/ensemblegpuarray/) and [CUDA sparse linear algebra documentation](https://cuda.juliagpu.org/stable/lib/cusparse/).

On the current 240-asset, six-productivity-state calibration, CPU sparse solves were faster than CUDA sparse QR: a representative 1,440-state system took about 0.8 ms on CPU and 26 ms on GPU including transfers on an NVIDIA L40S. The default therefore remains :cpu. The Broyden equilibrium search reduced complete lifecycle evaluations from 95 to 6 and elapsed solve time from about 115 seconds to 16 seconds on this machine, while preserving the equilibrium prices to numerical tolerance. GPU mode is available for experimentation with substantially larger systems:

    result = solve_stationary_equilibrium(mp; backend=:gpu)

The result reports the selected backend and equilibrium_evaluations. These timings exclude SVG generation and depend on hardware and Julia compilation state.

## SVG output

The plotting function writes the following files to figure/:

- extension_value_functions.svg compares working and retired values against assets for each productivity state at the midpoint of the retirement window.
- extension_consumption_policy.svg shows working consumption by productivity state and retired consumption. Gray worker cells are states where the household retires immediately; these do not have an active working policy.
- extension_saving_policy.svg shows asset drift, $ra+\text{income}-c$, using the same status convention.
- extension_retirement_region.svg shows continuation versus retirement across age and assets for every productivity state during the eligibility window.
- extension_stationary_distribution.svg shows worker, retiree, and total population density on the age–asset grid. Node masses are divided by their local asset-cell widths, so colors remain comparable across the nonuniform grid. The color scale is logarithmic; densities below $10^{-10}$ per year per million NTD are displayed at the floor.
- extension_asset_distributions.svg compares the conditional asset distributions of workers in each productivity state, retirees, and all modeled people combined. Age-specific masses are weighted by the stationary age population, then each group is normalized to unit mass. The left panel divides node probabilities by local asset-cell widths to show density on the nonuniform grid and displays the 0–0.2 density range; the right panel shows cumulative shares. The overall group includes workers and retirees.

The heatmaps are raster layers inside SVG files; axes, labels, and value-function lines remain vector graphics. All monetary axes use millions of NTD. Age is measured in years.
