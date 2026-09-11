using LinearAlgebra, SparseArrays, Statistics, HighVoronoi, QuasiMonteCarlo

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
    data_voronoi_grid(X_path=nothing; para, npoints=2048, stencil_scale=0.5)

Draw a uniform two-dimensional Sobol design with QuasiMonteCarlo.jl and scale
it directly from `[0,1]^2` to `[0,B_max] × [0,Q_max]`. The result contains
exactly `npoints` scattered states, not a tensor product. Use `npoints=2^k` to
retain the balance properties of a Sobol net. `X_path` is accepted only for
compatibility with earlier data-adapted versions and does not affect the nodes.

HighVoronoi supplies cell volumes and the Delaunay dual used for piecewise
linear interpolation. Boundary interpolation sites are auxiliary, not states.
For B-only diffusion on this unstructured mesh, directional wide stencils
replace the axis-aligned two-point formula. `stencil_scale` controls their
normalized reach, proportional to npoints^(-1/4); interpolation introduces
numerical diffusion at finite resolution. Refine nodes AND stencil reach when
checking accuracy. No exact rank-one covariance is claimed on a finite mesh.
"""
function data_voronoi_grid(X_path=nothing; para,npoints=2048,stencil_scale=0.5)
    npoints isa Integer && npoints >= 16 || throw(ArgumentError("npoints must be an integer at least 16"))
    isfinite(stencil_scale) && 0 < stencil_scale <= 1 || throw(ArgumentError("invalid stencil_scale"))
    limits = [Float64(para.B_max),Float64(para.Q_max)]
    all(x -> isfinite(x) && x>0,limits) || throw(ArgumentError("invalid domain"))
    design = QuasiMonteCarlo.sample(npoints,2,SobolSample())
    all(x -> isfinite(x) && 0 <= x <= 1,design) || throw(ArgumentError("sampler must return points in [0,1]^2"))
    points = clamp.(design,eps(),1-eps())
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
    return (;geometry,data,nodes=data.nodes,volumes,points,design,mesh,stencils,distances,
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
