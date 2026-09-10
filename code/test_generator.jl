using Test
include("generator.jl")

@testset "Joint QMC Voronoi generator" begin
    para = (B_max=1.0,Q_max=1.0,μB=(B,Q,V)->0.0,μQ=(Q,V)->0.5-Q,σB=B->0.1)
    path = [[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8]]
    grid = data_voronoi_grid(path;para,npoints=128,bandwidth=0.03)
    N = length(grid.nodes)
    @test N == 128
    @test grid.design == QuasiMonteCarlo.sample(128,2,SobolSample())
    @test all(x -> 0<x<1,grid.points)
    @test length(unique(first.(grid.nodes)))*length(unique(last.(grid.nodes))) > N
    @test sum(grid.volumes) ≈ 1
    for B in range(0,1;length=17), Q in range(0,1;length=17)
        _,weights = scattered_weights(grid.mesh,B,Q)
        @test minimum(weights) >= 0 && sum(weights) ≈ 1
    end
    for i in 1:N
        B,Q = grid.points[:,i]
        @test grid.density.B_cdf(B) ≈ grid.design[1,i] atol=1e-12
        @test grid.density.conditional_Q_cdf(B)(Q) ≈ grid.design[2,i] atol=1e-12
        owners,weights = scattered_weights(grid.mesh,B,Q)
        @test sum(weights) ≈ 1
        @test sum(w for (j,w) in zip(owners,weights) if j==i) ≈ 1 atol=1e-9
    end
    F = construct_forward_generator(0.0;para,grid)
    @test all(>=(0),nonzeros(F-spdiagm(0=>diag(F))))
    @test maximum(vec(sum(F;dims=1))) < 1e-10
    @test minimum(vec(sum(F;dims=1))) < 0
    @test minimum(exp(Matrix(F)*0.01)) >= -1e-12
    G = construct_forward_generator(0.0;para,grid,representation=:density)
    M = spdiagm(0=>grid.volumes)
    @test M*G ≈ F*M
    @test construct_forward_generator((B,Q)->0.0;para,grid) ≈ F
    @test construct_forward_generator(zeros(N);para,grid) ≈ F
    @test_throws DimensionMismatch construct_forward_generator(zeros(2,N);para,grid)
    @test_throws ArgumentError construct_forward_generator(0.0;para=merge(para,(μQ=(Q,V)->1.0,)),grid)
    @test_throws ArgumentError data_voronoi_grid(path;para,npoints=2)
    @test_throws ArgumentError data_voronoi_grid(path;para,bandwidth=0.0)
    @test_throws ArgumentError data_voronoi_grid(path;para,boundary_weight=1.0)
    @test_throws ArgumentError data_voronoi_grid([[-1.,0.]];para)

    # The joint transform retains dependence; the old product of marginals did not.
    correlated = data_voronoi_grid(path;para,npoints=256,bandwidth=0.02,
        data_weight=0.95,boundary_weight=0.0)
    @test cor(first.(correlated.nodes),last.(correlated.nodes)) > 0.8
    uniform = data_voronoi_grid(path;para,npoints=64,sampler=HaltonSample(),
        data_weight=0.0,boundary_weight=0.0)
    @test uniform.points ≈ QuasiMonteCarlo.sample(64,2,HaltonSample())
    # In the interior, directional interpolation reproduces affine functions.
    interior = findall(i -> all(x -> 0.3<x<0.7,uniform.points[:,i]),1:64)
    @test !isempty(interior)
    driftpara = merge(para,(μB=(B,Q,V)->1.0,μQ=(Q,V)->0.0,σB=B->0.0))
    A = adjoint(construct_forward_generator(0.0;para=driftpara,grid=uniform))
    @test maximum(abs,(A*first.(uniform.nodes))[interior] .- 1) < 1e-10
    @test maximum(abs,(A*last.(uniform.nodes))[interior]) < 1e-10
    diffusionpara = merge(para,(μQ=(Q,V)->0.0,σB=B->1.0))
    A = adjoint(construct_forward_generator(0.0;para=diffusionpara,grid=uniform))
    @test maximum(abs,(A*first.(uniform.nodes))[interior]) < 1e-10
    @test maximum(abs,(A*last.(uniform.nodes))[interior]) < 1e-10
    repeated = data_voronoi_grid(hcat(path...);para,npoints=64,sampler=HaltonSample(),
        data_weight=0.0,boundary_weight=0.0)
    @test uniform.points == repeated.points
    # Reflection endpoints are identity and lower-B endpoints are killing states.
    for i in 1:N, d in 1:4
        owners,weights = grid.stencils[i,d]
        @test all(>=(0),weights) && sum(weights) ≈ 1
        @test all(j -> 0 <= j <= N,owners)
    end
end
