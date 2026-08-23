using FastChebInterp
using LinearAlgebra
using Printf

export collocation_diff_matrix, cheb_diff_matrix

"""
Build a Chebyshev collocation differentiation matrix for a 1D grid.

The returned matrix satisfies D * f_vals ≈ df/dx at the collocation points.
"""
function cheb_diff_matrix(points::AbstractVector, lb, ub)
    n = length(points) - 1
    basis = Matrix{Float64}(I, n + 1, n + 1)
    D = zeros(Float64, n + 1, n + 1)

    for j in 1:(n + 1)
        interp = chebinterp(@view(basis[:, j]), lb, ub)
        D[:, j] = [getindex(chebgradient(interp, pt), 2) for pt in points]
    end

    return D
end

"""
Build differentiation matrices for tensor-product collocation values.

The input may be either:
- a 1D vector of collocation points, or
- the multidimensional output of chebpoints, i.e. an array of SVector points.

The returned tuple contains one full-space differentiation matrix per dimension.
These matrices act on the flattened collocation values in the same ordering as
`vec(points)`.
"""
function collocation_diff_matrix(points, lbs, ubs)
    if points isa AbstractVector && !(eltype(points) <: AbstractVector)
        return (cheb_diff_matrix(points, lbs, ubs),)
    end

    if points isa Tuple
        return tuple([cheb_diff_matrix(p, lb, ub) for (p, lb, ub) in zip(points, lbs, ubs)]...)
    end

    pts = collect(points)
    if isempty(pts)
        throw(ArgumentError("points must not be empty"))
    end

    if pts[1] isa Number
        return (cheb_diff_matrix(pts, lbs, ubs),)
    end

    dims = size(points)
    coord_sets = [unique([p[k] for p in pts]) for k in 1:length(dims)]

    Dmats = Vector{Matrix{Float64}}(undef, length(dims))
    for k in 1:length(dims)
        D1d = cheb_diff_matrix(coord_sets[k], lbs[k], ubs[k])
        M = Matrix{Float64}(I, 1, 1)
        for j in reverse(1:length(dims))
            factor = j == k ? D1d : Matrix{Float64}(I, dims[j], dims[j])
            M = kron(M, factor)
        end
        Dmats[k] = M
    end

    return tuple(Dmats...)
end