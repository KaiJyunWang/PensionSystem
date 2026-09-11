using Test
include("stochastic_voronoi_run.jl")

@testset "Scattered Voronoi pension solver" begin
    para = voronoi_model()
    observations = load_voronoi_data()
    @test first(observations.sample.date) == Date(2009,2,1)
    @test last(observations.sample.date) == Date(2019,12,1)
    @test all(<(Date(2020,1,1)),observations.sample.date)
    grid = pension_voronoi_grid(observations.X_path;para,npoints=128)
    @test grid.points == QuasiMonteCarlo.sample(128,2,SobolSample())
    result = solve_voronoi_pde(;para,grid,verbose=false)
    @test result.residual <= 1e-8
    @test minimum(result.values) >= 0
    @test maximum(result.values) <= para.p/(para.ρ+para.m)+1e-8
    @test result.A ≈ adjoint(result.F)
    G = construct_forward_generator(result.values;para,grid,representation=:density)
    M = spdiagm(0=>grid.volumes)
    @test M*result.A ≈ adjoint(G)*M
    @test norm((para.ρ+para.m)*result.values-result.A*result.values .- para.p,Inf) <= 1e-8
    @test all(result.V(B,Q) ≈ result.values[i] for (i,(B,Q)) in enumerate(grid.nodes))
    for Q in (0.0,grid.Q_max/2,grid.Q_max)
        @test result.V(0.0,Q) == 0
        @test result.V(2grid.B_max,Q) ≈ result.V(grid.B_max,Q)
        @test result.V(-0.002,Q) ≈ 2result.V(-0.001,Q)
    end
    for B in (-0.1,grid.B_max/2,2grid.B_max)
        @test result.V(B,-grid.Q_max) ≈ result.V(B,0.0)
        @test result.V(B,2grid.Q_max) ≈ result.V(B,grid.Q_max)
    end
    @test_throws ErrorException solve_voronoi_pde(;para,grid,iterations=1,verbose=false)
    mktempdir() do dir
        df = CSV.read(joinpath(@__DIR__,"..","data","labor_insurance_fund_monthly.csv"),DataFrame)
        dates = [x isa Date ? x : Date(string(x),dateformat"yyyy-mm") for x in df.date]
        df[dates .>= Date(2020,1,1),:fund_level_ntd] .= 0
        file = joinpath(dir,"changed.csv")
        CSV.write(file,df)
        @test load_voronoi_data(;path=file).X_path == observations.X_path
        run = run_voronoi_pension(;para,grid_options=(npoints=128,),solver_options=(verbose=false,),output_dir=dir)
        @test isfile(joinpath(dir,"voronoi_value.csv"))
        @test nrow(CSV.read(joinpath(dir,"voronoi_prefinance.csv"),DataFrame)) == length(observations.X_path)
        @test run.solution.residual <= 1e-8
    end
end
