using Test
include("generator.jl")

@testset "Data-adapted Voronoi forward generator" begin
    para = (B_max=1.0, Q_max=1.0, μB=(B,Q,V)->0.0,
            μQ=(Q,V)->0.5-Q, σB=B->0.1)
    observations = [[0.4, 0.3], [0.6, 0.7]]
    grid = data_voronoi_grid(observations; para, nB=9, nQ=9)
    N = length(grid.nodes)
    @test sum(grid.volumes) ≈ 1.0
    @test length(grid.Bs) == length(grid.Qs) == 9
    @test all(0 .< grid.Bs .< para.B_max)
    @test all(0 .< grid.Qs .< para.Q_max)
    F = construct_forward_generator(0.0; para, grid)
    @test size(F) == (N, N)
    @test all(>=(0), nonzeros(F - spdiagm(0 => diag(F))))
    sums = vec(sum(F; dims=1))
    @test maximum(sums) < 1e-10
    for i in 1:N
        touches_bottom = N + 2 in grid.data.neighbors[i]
        @test touches_bottom ? sums[i] < 0 : abs(sums[i]) < 1e-10
    end
    density = construct_forward_generator(0.0; para, grid, representation=:density)
    @test spdiagm(0 => grid.volumes) * density ≈ F * spdiagm(0 => grid.volumes)
    @test construct_forward_generator((B,Q)->0.0; para, grid) ≈ F
    @test construct_forward_generator(zeros(length(grid.Bs), length(grid.Qs)); para, grid) ≈ F
    @test_throws DimensionMismatch construct_forward_generator(zeros(N+1); para, grid)
    @test_throws ArgumentError construct_forward_generator(0.0; para=merge(para, (μQ=(Q,V)->1.0,)), grid)
    @test_throws ArgumentError data_voronoi_grid([[-0.1,0.5]]; para)

    # Pure B drift travels only in the upwind direction. Upper B reflects;
    # lower B loses mass only for outward drift.
    for direction in (-1.0, 1.0)
        driftpara = merge(para, (μB=(B,Q,V)->direction, μQ=(Q,V)->0.0, σB=B->0.0))
        A = construct_forward_generator(0.0; para=driftpara, grid)
        rows, cols, vals = findnz(A)
        for (i,j,a) in zip(rows,cols,vals)
            if i != j && a > 0
                @test direction * (grid.nodes[i][1] - grid.nodes[j][1]) > 0
                @test grid.nodes[i][2] == grid.nodes[j][2]
            end
        end
        direction > 0 && @test maximum(abs, vec(sum(A; dims=1))) < 1e-10
    end
    # Diffusion in B must never produce transitions in Q.
    diffusionpara = merge(para, (μQ=(Q,V)->0.0,))
    A = construct_forward_generator(0.0; para=diffusionpara, grid)
    rows, cols, vals = findnz(A)
    @test all(i == j || a == 0 || grid.nodes[i][2] == grid.nodes[j][2]
              for (i,j,a) in zip(rows,cols,vals))
    @test minimum(exp(Matrix(F) * 0.01)) >= -1e-12
end


@testset "Smooth QMC node density" begin
    @test [radical_inverse(k,2) for k in 1:4] == [0.5,0.25,0.75,0.125]
    @test [radical_inverse(k,3) for k in 1:3] ≈ [1/3,2/3,1/9]
    density = smooth_node_density([0.2,0.3]; bandwidth=0.04,
        data_weight=0.6, boundary_weight=0.25, boundary_scale=0.02)
    @test density.cdf(0) == 0 && density.cdf(1) == 1
    xs = range(0,1; length=10001)
    ys = density.pdf.(xs)
    @test (sum(ys) - (ys[1]+ys[end])/2) * step(xs) ≈ 1 atol=2e-6
    @test all(>(0),diff(density.cdf.(xs)))
    @test minimum(ys) >= 0.15-1e-12
    @test density.pdf(0) > density.pdf(0.8)
    @test density.pdf(0.25) > density.pdf(0.8)
    for x in (0.1,0.25,0.5)
        @test (density.cdf(x+1e-6)-density.cdf(x-1e-6))/2e-6 ≈ density.pdf(x) rtol=1e-7
    end
    for u in (1e-5,0.1,0.5,0.9,1-1e-5)
        @test density.cdf(density_quantile(density.cdf,u)) ≈ u atol=1e-12
    end
    para = (B_max=1.0,Q_max=1.0)
    path = [[0.2,0.3],[0.25,0.4]]
    grid = data_voronoi_grid(path; para,nB=16,nQ=12)
    @test sort(grid.densities[1].cdf.(grid.Bs)) ≈ sort([radical_inverse(k,2) for k in 1:16])
    @test sort(grid.densities[2].cdf.(grid.Qs)) ≈ sort([radical_inverse(k,3) for k in 1:12])
    @test count(<(0.1),grid.Bs) >= 3
    @test count(x -> 0.15 < x < 0.3,grid.Bs) >= 8
    @test maximum(grid.Bs) > 0.5
    repeat_grid = data_voronoi_grid(hcat(path...); para,nB=16,nQ=12)
    @test grid.Bs == repeat_grid.Bs && grid.Qs == repeat_grid.Qs
    uniform = data_voronoi_grid(path; para,nB=4,nQ=4,data_weight=0.0,boundary_weight=0.0)
    @test uniform.Bs ≈ sort([radical_inverse(k,2) for k in 1:4])
    shifted = data_voronoi_grid(path; para,nB=4,nQ=4,data_weight=0.0,boundary_weight=0.0,qmc_skip=4)
    @test shifted.Bs ≈ sort([radical_inverse(k,2) for k in 5:8])
    @test_throws ArgumentError data_voronoi_grid(path; para,bandwidth=0.0)
    @test_throws ArgumentError data_voronoi_grid(path; para,boundary_scale=0.0)
    @test_throws ArgumentError data_voronoi_grid(path; para,data_weight=0.8,boundary_weight=0.3)
    @test_throws ArgumentError data_voronoi_grid(path; para,nB=1)
    @test_throws ArgumentError data_voronoi_grid(path; para,qmc_skip=-1)
    # Repeated observations and observations exactly on the domain boundary
    # still produce a smooth, normalized density, without clipping nodes.
    edge = smooth_node_density([0.0,0.0,1.0]; bandwidth=0.02,
        data_weight=0.6,boundary_weight=0.25,boundary_scale=0.005)
    @test isfinite(edge.pdf(0.0)) && isfinite(edge.pdf(1.0))
    @test 0 < density_quantile(edge.cdf,0.01) < density_quantile(edge.cdf,0.99) < 1
end
