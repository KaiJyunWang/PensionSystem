using LinearAlgebra, SparseArrays, Statistics, HighVoronoi, QuasiMonteCarlo

# Stable logistic CDF, avoiding a new dependency in the generator helper.
_logistic_cdf(x) = x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x))

"""
    smooth_node_density(samples; bandwidth, data_weight, boundary_weight, boundary_scale)

Continuous density on normalized [0,1]: a mixture of equally weighted,
individually truncated logistic kernels at the observations, a truncated
exponential concentrated at zero, and a uniform coverage component.
Returns `pdf` and `cdf`. No clipping or rejection sampling is used.
"""
function smooth_node_density(samples; bandwidth, data_weight, boundary_weight, boundary_scale)
    isempty(samples) && throw(ArgumentError("density needs observations"))
    all(x -> isfinite(x) && 0 <= x <= 1, samples) || throw(ArgumentError("invalid normalized observations"))
    isfinite(bandwidth) && 1e-8 <= bandwidth <= 1 || throw(ArgumentError("bandwidth must be in [1e-8,1]"))
    isfinite(boundary_scale) && 1e-8 <= boundary_scale <= 1 || throw(ArgumentError("boundary_scale must be in [1e-8,1]"))
    all(isfinite, (data_weight, boundary_weight)) && data_weight >= 0 && boundary_weight >= 0 &&
        data_weight + boundary_weight <= 1 || throw(ArgumentError("weights must be nonnegative and sum to at most one"))
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
Joint bounded KDE plus a lower-B exponential layer and uniform coverage.
The Rosenblatt map first inverts the B marginal, then the conditional Q CDF.
Each KDE component is centered on an observed PAIR, preserving dependence.
All scales refer to normalized coordinates in [0,1]^2.
"""
function joint_node_density(observations; bandwidth, data_weight, boundary_weight, boundary_scale)
    data_weight + boundary_weight < 1 || throw(ArgumentError("joint density needs a positive uniform coverage weight"))
    marginal = smooth_node_density(first.(observations); bandwidth=bandwidth[1],
        data_weight, boundary_weight, boundary_scale)
    kernels = [ntuple(d -> smooth_node_density([x[d]]; bandwidth=bandwidth[d],
        data_weight=1.0, boundary_weight=0.0, boundary_scale), 2) for x in observations]
    kernel_pdf(k,x) = k.pdf(x)
    kernel_cdf(k,x) = k.cdf(x)
    function conditional(B)
        weights = [data_weight/length(kernels)*kernel_pdf(k[1],B) for k in kernels]
        uniform = 1-data_weight-boundary_weight + boundary_weight*exp(-B/boundary_scale)/
                  (boundary_scale*(-expm1(-1/boundary_scale)))
        normalizer = uniform + sum(weights)
        cdf(Q) = (uniform*Q + sum(w*kernel_cdf(k[2],Q) for (w,k) in zip(weights,kernels)))/normalizer
        return cdf
    end
    function pdf(B,Q)
        (0 <= B <= 1 && 0 <= Q <= 1) || return 0.0
        return 1-data_weight-boundary_weight +
            boundary_weight*exp(-B/boundary_scale)/(boundary_scale*(-expm1(-1/boundary_scale))) +
            data_weight/length(kernels)*sum(kernel_pdf(k[1],B)*kernel_pdf(k[2],Q) for k in kernels)
    end
    return (; pdf, B_cdf=marginal.cdf, conditional_Q_cdf=conditional)
end

# Build the Delaunay dual of a HighVoronoi mesh for barycentric interpolation.
# Auxiliary edge sites close the rectangular convex hull; they are not states.
function scattered_interpolation_mesh(points; nedge)
    N = size(points,2)
    sites = [collect(x) for x in eachcol(points)]
    owners = collect(1:N)
    for t in range(0,1;length=nedge), (B,Q) in ((0.0,t),(1.0,t),(t,0.0),(t,1.0))
        x = [B,Q]
        x in sites && continue
        push!(sites,x)
        # Homogeneous lower Dirichlet. Other edges extend nearby nodal values.
        owner = B == 0 ? 0 : argmin([sum(abs2,x-points[:,i]) for i in 1:N])
        push!(owners,owner)
    end
    # An unbounded construction retains circumcenters far outside the physical
    # domain, which are common for thin triangles adjacent to B=0.
    vg = VoronoiGeometry(VoronoiNodes(hcat(sites...));integrate=false,silence=true)
    vd = VoronoiData(vg;getvertices=true)
    polygons = Set{Tuple{Vararg{Int}}}()
    for cell in vd.vertices, (signature, _) in cell
        all(j -> 1 <= j <= length(sites),signature) || continue
        length(signature) >= 3 && push!(polygons,Tuple(sort(collect(signature))))
    end
    triangles = NTuple{3,Int}[]
    for polygon in sort!(collect(polygons))
        center = sum(sites[j] for j in polygon)/length(polygon)
        order = sort(collect(polygon);by=j -> atan(sites[j][2]-center[2],sites[j][1]-center[1]))
        for k in 2:length(order)-1
            push!(triangles,(order[1],order[k],order[k+1]))
        end
    end
    isempty(triangles) && error("HighVoronoi returned no interpolation triangles")
    # Spatial bins make point location inexpensive during plotting and assembly.
    nbins = max(2,ceil(Int,sqrt(N/2)))
    bins = [Int[] for _ in 1:nbins, _ in 1:nbins]
    bin(x) = clamp(floor(Int,x*nbins)+1,1,nbins)
    for (t,tri) in enumerate(triangles)
        lo = [minimum(sites[j][d] for j in tri) for d in 1:2]
        hi = [maximum(sites[j][d] for j in tri) for d in 1:2]
        for i in bin(lo[1]):bin(hi[1]), j in bin(lo[2]):bin(hi[2])
            push!(bins[i,j],t)
        end
    end
    return (; sites,owners,triangles,bins,nbins)
end

function scattered_weights(mesh,B,Q)
    0 <= B <= 1 && 0 <= Q <= 1 || throw(ArgumentError("interpolation point outside normalized domain"))
    bin(x) = clamp(floor(Int,x*mesh.nbins)+1,1,mesh.nbins)
    for t in mesh.bins[bin(B),bin(Q)]
        tri = mesh.triangles[t]
        a,b,c = (mesh.sites[j] for j in tri)
        det = (b[1]-a[1])*(c[2]-a[2]) - (b[2]-a[2])*(c[1]-a[1])
        abs(det) > 1e-18 || continue
        v = ((B-a[1])*(c[2]-a[2])-(Q-a[2])*(c[1]-a[1]))/det
        w = ((b[1]-a[1])*(Q-a[2])-(b[2]-a[2])*(B-a[1]))/det
        weights = [1-v-w,v,w]
        minimum(weights) >= -1e-10 || continue
        weights = max.(weights,0)
        weights ./= sum(weights)
        return (ntuple(k -> mesh.owners[tri[k]],3),Tuple(weights))
    end
    error("could not locate interpolation triangle at ($B,$Q)")
end

"""
    data_voronoi_grid(X_path; para, npoints=2048, sampler=SobolSample(),
        bandwidth=(0.002,0.02), data_weight=0.6, boundary_weight=0.25,
        boundary_scale=0.005, stencil_scale=0.5)

Draw a 2×npoints design with QuasiMonteCarlo.jl and map each column through
inverse B and conditional-Q CDFs of a smooth JOINT density. The result contains
exactly npoints scattered states, not a tensor product. Use npoints=2^k for
Sobol designs. `sampler` can also be `HaltonSample()` or a randomized sampler.
Zero/one sampler endpoints are moved by machine epsilon into the interior.

HighVoronoi supplies cell volumes and the Delaunay dual used for piecewise
linear interpolation. Boundary interpolation sites are auxiliary, not states.
For B-only diffusion on this unstructured mesh, directional wide stencils
replace the axis-aligned two-point formula. `stencil_scale` controls their
normalized reach, proportional to npoints^(-1/4); interpolation introduces
numerical diffusion at finite resolution. Refine nodes AND stencil reach when
checking accuracy. No exact rank-one covariance is claimed on a finite mesh.
"""
function data_voronoi_grid(X_path; para,npoints=2048,sampler=SobolSample(),
        bandwidth=(0.002,0.02),data_weight=0.6,boundary_weight=0.25,
        boundary_scale=0.005,stencil_scale=0.5)
    npoints isa Integer && npoints >= 16 || throw(ArgumentError("npoints must be an integer at least 16"))
    isfinite(stencil_scale) && 0 < stencil_scale <= 1 || throw(ArgumentError("invalid stencil_scale"))
    observations = X_path isa AbstractMatrix ? collect(eachcol(X_path)) : collect(X_path)
    isempty(observations) && throw(ArgumentError("X_path is empty"))
    all(x -> length(x)==2 && all(isfinite,x),observations) || throw(ArgumentError("invalid observations"))
    limits = [Float64(para.B_max),Float64(para.Q_max)]
    all(x -> isfinite(x) && x>0,limits) || throw(ArgumentError("invalid domain"))
    all(x -> all(0 <= x[d] <= limits[d] for d in 1:2),observations) || throw(ArgumentError("observations outside domain"))
    widths = bandwidth isa Real ? (bandwidth,bandwidth) : bandwidth
    length(widths)==2 || throw(ArgumentError("bandwidth must have two components"))
    density = joint_node_density([x./limits for x in observations];bandwidth=widths,
        data_weight,boundary_weight,boundary_scale)
    design = QuasiMonteCarlo.sample(npoints,2,sampler)
    all(x -> isfinite(x) && 0 <= x <= 1,design) || throw(ArgumentError("sampler must return points in [0,1]^2"))
    points = zeros(2,npoints)
    for i in 1:npoints
        B = density_quantile(density.B_cdf,clamp(design[1,i],eps(),1-eps()))
        Q = density_quantile(density.conditional_Q_cdf(B),clamp(design[2,i],eps(),1-eps()))
        points[:,i] = [B,Q]
    end
    length(Set(Tuple.(eachcol(points)))) == npoints || throw(ArgumentError("sampler produced duplicate nodes"))
    physical = points .* limits
    geometry = VoronoiGeometry(VoronoiNodes(physical),
        cuboid(2;dimensions=limits,periodic=Int[],neumann=[1,2,-2]);
        integrator=HighVoronoi.VI_POLYGON,silence=true)
    data = VoronoiData(geometry)
    volumes = Float64.(collect(data.volume))
    all(x -> isfinite(x) && x>0,volumes) || error("invalid cell volumes")
    mesh = scattered_interpolation_mesh(points;nedge=max(8,ceil(Int,sqrt(npoints))))
    reach = stencil_scale*npoints^(-1/4)
    distances = zeros(npoints,4)
    stencils = Matrix{Tuple{NTuple{3,Int},NTuple{3,Float64}}}(undef,npoints,4)
    for i in 1:npoints, direction in 1:4
        d = direction <= 2 ? 1 : 2
        sign = isodd(direction) ? 1 : -1
        x = copy(points[:,i])
        boundary_distance = sign > 0 ? 1-x[d] : x[d]
        h = min(reach,boundary_distance)
        touches_boundary = h == boundary_distance
        x[d] = touches_boundary ? (sign > 0 ? 1.0 : 0.0) : x[d]+sign*h
        distances[i,direction] = h*limits[d]
        stencils[i,direction] = if touches_boundary
            owner = d == 1 && x[d] == 0 ? 0 : i
            ((owner,owner,owner),(1.0,0.0,0.0))
        else
            scattered_weights(mesh,x...)
        end
    end
    return (;geometry,data,nodes=data.nodes,volumes,points,design,density,mesh,stencils,distances,
        B_max=limits[1],Q_max=limits[2])
end

"""
Forward mass generator F, with backward adjoint A=F'. Positive drift uses
forward directional interpolation; negative drift uses backward interpolation.
B diffusion uses a nonuniform three-point directional stencil. B=0 is killed;
other boundaries have no exterior transitions. Q boundary drift must be inward.
V must be a scalar, callable, or vector in grid.nodes order. For density,
returns M^-1 F M. Interpolation weights keep off-diagonal rates nonnegative.
"""
function construct_forward_generator(V;para,grid,representation=:mass)
    representation in (:mass,:density) || throw(ArgumentError("invalid representation"))
    N = length(grid.nodes)
    if V isa AbstractArray
        V isa AbstractVector && length(V)==N || throw(DimensionMismatch("V must be a nodal vector"))
    end
    value(i,B,Q) = V isa Number ? V : V isa AbstractVector ? V[i] : V(B,Q)
    rows,cols,rates = Int[],Int[],Float64[]
    for i in 1:N
        B,Q = grid.nodes[i]
        v = value(i,B,Q)
        b,q,D = para.μB(B,Q,v),para.μQ(Q,v),0.5*para.σB(B)^2
        all(isfinite,(v,b,q,D)) || throw(ArgumentError("nonfinite coefficients"))
        qlo,qhi = para.μQ(0.0,value(i,B,0.0)),para.μQ(grid.Q_max,value(i,B,grid.Q_max))
        isfinite(qlo) && isfinite(qhi) && qlo >= 0 && qhi <= 0 || throw(ArgumentError("Q boundary drift must be inward"))
        hp,hm,hqp,hqm = grid.distances[i,:]
        coefficients = (max(b,0)/hp + 2D/(hp*(hp+hm)),
                        max(-b,0)/hm + 2D/(hm*(hp+hm)),max(q,0)/hqp,max(-q,0)/hqm)
        total = 0.0
        for d in 1:4
            coefficient = coefficients[d]
            isfinite(coefficient) || throw(ArgumentError("nonfinite transition rate; nodes too close to boundary"))
            owners,weights = grid.stencils[i,d]
            all(==(i),owners) && continue # Reflecting endpoint: f(endpoint)=f(i).
            for (j,w) in zip(owners,weights)
                j == 0 && continue
                push!(rows,j);push!(cols,i);push!(rates,coefficient*w)
            end
            total += coefficient
        end
        push!(rows,i);push!(cols,i);push!(rates,-total)
    end
    F = sparse(rows,cols,rates,N,N)
    return representation == :mass ? F : spdiagm(0=>1 ./ grid.volumes)*F*spdiagm(0=>grid.volumes)
end
