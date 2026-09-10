using LinearAlgebra, SparseArrays, Statistics, HighVoronoi

# Radical-inverse coordinates (the one-dimensional components of Halton).
# A tensor product of these sequences keeps the Voronoi faces axis aligned.
function radical_inverse(index::Integer, base::Integer)
    index >= 1 && base >= 2 || throw(ArgumentError("invalid radical-inverse index or base"))
    value, factor = 0.0, 1.0 / base
    while index > 0
        index, digit = divrem(index, base)
        value += digit * factor
        factor /= base
    end
    return value
end

# Stable logistic CDF, avoiding a new dependency in the generator helper.
_logistic_cdf(x) = x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x))

"""
    smooth_node_density(samples; bandwidth, data_weight, boundary_weight, boundary_scale)

Continuous density on normalized [0,1]: a mixture of equally weighted,
individually truncated logistic kernels at the observations, a truncated
exponential concentrated at zero, and a positive uniform coverage component.
Returns `pdf` and `cdf`. No clipping or rejection sampling is used.
"""
function smooth_node_density(samples; bandwidth, data_weight, boundary_weight, boundary_scale)
    isempty(samples) && throw(ArgumentError("density needs observations"))
    all(x -> isfinite(x) && 0 <= x <= 1, samples) || throw(ArgumentError("invalid normalized observations"))
    isfinite(bandwidth) && 1e-8 <= bandwidth <= 1 || throw(ArgumentError("bandwidth must be in [1e-8,1]"))
    isfinite(boundary_scale) && 1e-8 <= boundary_scale <= 1 || throw(ArgumentError("boundary_scale must be in [1e-8,1]"))
    all(isfinite, (data_weight, boundary_weight)) && data_weight >= 0 && boundary_weight >= 0 &&
        data_weight + boundary_weight < 1 || throw(ArgumentError("weights must be nonnegative and sum to less than one"))
    lower = [_logistic_cdf(-c / bandwidth) for c in samples]
    normalizers = [_logistic_cdf((1-c) / bandwidth) - lo for (c,lo) in zip(samples,lower)]
    coverage_weight = 1 - data_weight - boundary_weight
    exponential_norm = -expm1(-1 / boundary_scale)
    function cdf(x)
        x <= 0 && return 0.0
        x >= 1 && return 1.0
        kernel = sum((_logistic_cdf((x-c)/bandwidth)-lo)/z
                     for (c,lo,z) in zip(samples,lower,normalizers)) / length(samples)
        return coverage_weight*x + data_weight*kernel +
               boundary_weight*(-expm1(-x/boundary_scale))/exponential_norm
    end
    function pdf(x)
        (x < 0 || x > 1) && return 0.0
        kernel = sum(begin
            t = exp(-abs((x-c)/bandwidth))
            t / ((1+t)^2 * bandwidth * z)
        end for (c,z) in zip(samples,normalizers)) / length(samples)
        return coverage_weight + data_weight*kernel +
               boundary_weight*exp(-x/boundary_scale)/(boundary_scale*exponential_norm)
    end
    return (; pdf, cdf)
end

# Monotone inverse CDF: the uniform mixture makes the inverse unique.
function density_quantile(cdf, u)
    0 < u < 1 || throw(ArgumentError("quantile probability must be strictly interior"))
    lo, hi = 0.0, 1.0
    for _ in 1:60
        mid = (lo + hi) / 2
        if cdf(mid) < u
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

"""
    data_voronoi_grid(X_path; para, nB=96, nQ=64, bandwidth=(0.002,0.02),
                      data_weight=0.6, boundary_weight=0.25,
                      boundary_scale=0.005, qmc_skip=0)

Sample nonuniform coordinates from smooth densities fitted to observations
`[[B,Q], ...]` (or a 2×N matrix) within the model bounds. Bandwidths and
`boundary_scale` are fractions of the corresponding domain length.

B density: `data_weight` times a truncated logistic KDE, `boundary_weight`
times a truncated exponential at zero, plus uniform coverage. Q density:
`data_weight` times its KDE plus uniform coverage. Increase bandwidth for a
more gradual transition around the data, or boundary_scale for a wider B layer.

Use inverse-CDF transforms of base-2 (B) and base-3 (Q) radical-inverse
sequences. Placement is deterministic, with exactly nB*nQ interior nodes;
qmc_skip advances each sequence. Density values and CDFs in normalized
coordinates are returned in `grid.densities` for inspection.

The tensor product is intentional: the current monotone B-only diffusion
requires axis-aligned faces. The sampling density is a product of marginals;
it does not preserve B–Q dependence as a joint KDE would. B varies fastest.
"""
function data_voronoi_grid(X_path; para, nB=96, nQ=64, bandwidth=(0.002,0.02),
        data_weight=0.6, boundary_weight=0.25, boundary_scale=0.005, qmc_skip=0)
    nB isa Integer && nQ isa Integer && nB >= 2 && nQ >= 2 ||
        throw(ArgumentError("nB and nQ must be integers at least two"))
    qmc_skip isa Integer && 0 <= qmc_skip <= typemax(Int)-max(nB,nQ) ||
        throw(ArgumentError("invalid qmc_skip"))
    observations = X_path isa AbstractMatrix ? collect(eachcol(X_path)) : collect(X_path)
    isempty(observations) && throw(ArgumentError("X_path is empty"))
    all(x -> length(x) == 2 && all(isfinite, x), observations) ||
        throw(ArgumentError("observations must be finite (B, Q) pairs"))
    limits = Float64[para.B_max, para.Q_max]
    all(x -> isfinite(x) && x > 0, limits) || throw(ArgumentError("invalid domain bounds"))
    all(x -> all(0 <= x[d] <= limits[d] for d in 1:2), observations) ||
        throw(ArgumentError("observations lie outside the domain"))
    widths = bandwidth isa Real ? (bandwidth, bandwidth) : bandwidth
    length(widths) == 2 || throw(ArgumentError("bandwidth needs one or two values"))
    densities = ntuple(d -> smooth_node_density([x[d]/limits[d] for x in observations];
        bandwidth=widths[d], data_weight, boundary_weight=d == 1 ? boundary_weight : 0.0,
        boundary_scale), 2)
    function sampled_axis(d, count, base)
        axis = sort!([limits[d] * density_quantile(densities[d].cdf,
                     radical_inverse(qmc_skip+k, base)) for k in 1:count])
        minimum(diff(axis)) > 1e-10*limits[d] ||
            throw(ArgumentError("sampled coordinates too close; increase bandwidth or boundary_scale"))
        return axis
    end
    Bs, Qs = sampled_axis(1,nB,2), sampled_axis(2,nQ,3)
    points = hcat(vec([[B, Q] for B in Bs, Q in Qs])...)
    boundary = cuboid(2; dimensions=limits, periodic=Int[], neumann=[1, 2, -2])
    geometry = VoronoiGeometry(VoronoiNodes(points), boundary;
        integrator=HighVoronoi.VI_POLYGON, silence=true)
    data = VoronoiData(geometry)
    volumes = Float64.(collect(data.volume))
    all(x -> isfinite(x) && x > 0, volumes) || error("invalid Voronoi cell volumes")
    return (; geometry, data, nodes=data.nodes, volumes, Bs, Qs, densities,
            B_max=limits[1], Q_max=limits[2])
end

"""
    construct_forward_generator(V; para, grid, representation=:mass)

Discretize `L f = μB*f_B + μQ*f_Q + σB^2/2*f_BB` on a grid from
`data_voronoi_grid`. `para` supplies `μB(B,Q,V)`, `μQ(Q,V)`, and `σB(B)`.
`V` is a scalar, callable `V(B,Q)`, vector in `grid.nodes` order, or matrix
of size `(length(grid.Bs), length(grid.Qs))`.

Positive drift uses the forward neighbor in the backward operator; negative
drift uses the backward neighbor. The returned sparse matrix is its forward
adjoint: `dm/dt = F*m` for cell masses. With `representation=:density`, it
instead evolves cell densities, with masses `grid.volumes .* density`.

Homogeneous Dirichlet at B=0 kills mass (no reinjection); homogeneous Neumann
at B=B_max reflects it. Q boundaries have no exterior flux. Their physical
drifts must be inward or zero; outward drift raises an error. Boundary drift
checks use adjacent nodal V, or boundary V when V is callable.

Example, using the model and observations from stochastic_fd.jl:
```julia
grid = data_voronoi_grid(X_path; para)
F = construct_forward_generator(V; para, grid)
# Evolve masses with expv(Δ, F, m0), without transposing F again.
```
"""
function construct_forward_generator(V; para, grid, representation=:mass)
    representation in (:mass, :density) || throw(ArgumentError("representation must be :mass or :density"))
    data, volumes = grid.data, grid.volumes
    N = length(grid.nodes)
    if V isa AbstractArray
        (V isa AbstractVector ? length(V) == N : size(V) == (length(grid.Bs), length(grid.Qs))) ||
            throw(DimensionMismatch("V must match the grid nodes"))
    end
    value(i, B, Q) = V isa Number ? V : V isa AbstractArray ? V[i] : V(B, Q)
    rows, cols, rates = Int[], Int[], Float64[]
    diagonal = zeros(N)
    for i in 1:N
        B, Q = grid.nodes[i]
        v = value(i, B, Q)
        b, q, D = para.μB(B, Q, v), para.μQ(Q, v), 0.5 * para.σB(B)^2
        all(isfinite, (v, b, q, D)) || throw(ArgumentError("nonfinite V or coefficients at node $i"))
        for (k, j) in enumerate(data.neighbors[i])
            area = data.area[i][k]
            area > 0 || continue
            if j <= N
                displacement = grid.nodes[j] - grid.nodes[i]
                distance = norm(displacement)
                normal = displacement / distance
                # Positive source-to-neighbor rates; diffusion crosses B faces only.
                abs(normal[1] * normal[2]) < 1e-8 ||
                    throw(ArgumentError("grid must have axis-aligned Voronoi faces"))
                rate = area / volumes[i] *
                    (max(b * normal[1] + q * normal[2], 0) + D * normal[1]^2 / distance)
                push!(rows, j); push!(cols, i); push!(rates, rate)
                diagonal[i] -= rate
            else
                # cuboid orders planes as B upper, B lower, Q upper, Q lower.
                plane = j - N
                if plane == 2
                    diagonal[i] -= area / volumes[i] * (max(-b, 0) + D / B)
                elseif plane == 3 || plane == 4
                    Qface = plane == 3 ? grid.Q_max : 0.0
                    qface = para.μQ(Qface, value(i, B, Qface))
                    isfinite(qface) || throw(ArgumentError("nonfinite Q boundary drift"))
                    inward = plane == 3 ? qface <= 0 : qface >= 0
                    inward || throw(ArgumentError("μQ must point inward at Q=$Qface (node $i)"))
                elseif plane != 1
                    throw(ArgumentError("unexpected boundary plane $plane"))
                end
                # Upper B: zero normal derivative, hence no exterior transition.
            end
        end
    end
    append!(rows, 1:N); append!(cols, 1:N); append!(rates, diagonal)
    F = sparse(rows, cols, rates, N, N)
    return representation == :mass ? F : spdiagm(0 => 1 ./ volumes) * F * spdiagm(0 => volumes)
end
