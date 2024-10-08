@testset "Polynomial" begin
  @testset "Basics" begin
    d = georef((; z=[1, 2, 3]))
    poly = GeoStatsModels.fit(Polynomial(), d)
    pred = GeoStatsModels.predict(poly, :z, Point(0.5))
    @test pred ≈ 1
    pred = GeoStatsModels.predict(poly, :z, Point(1.5))
    @test pred ≈ 2
    pred = GeoStatsModels.predict(poly, :z, Point(2.5))
    @test pred ≈ 3

    fitpredict(m, d) = GeoStatsModels.fitpredict(m, d, domain(d), neighbors=false)

    # constant trend
    rng = StableRNG(42)
    d = georef((z=rand(rng, 100),), CartesianGrid(100))
    z̄ = fitpredict(Polynomial(), d).z
    @test all(abs.(diff(z̄)) .< 0.01)

    # linear trend
    rng = StableRNG(42)
    μ = range(0, stop=1, length=100)
    ϵ = 0.1rand(rng, 100)
    d = georef((z=μ + ϵ,), CartesianGrid(100))
    z̄ = fitpredict(Polynomial(), d).z
    @test all([abs(z̄[i] - μ[i]) < 0.1 for i in 1:length(z̄)])

    # quadratic trend
    rng = StableRNG(42)
    r = range(-1, stop=1, length=100)
    μ = [x^2 + y^2 for x in r, y in r]
    ϵ = 0.1rand(rng, 100, 100)
    d = georef((z=μ + ϵ,))
    z̄ = fitpredict(Polynomial(2), d).z
    @test all([abs(z̄[i] - μ[i]) < 0.1 for i in 1:length(z̄)])

    # correct schema
    rng = StableRNG(42)
    d = georef((a=rand(rng, 10), b=rand(rng, 10)), rand(rng, Point, 10))
    d̄ = fitpredict(Polynomial(), d)
    t̄ = values(d̄)
    @test propertynames(t̄) == (:a, :b)
    @test eltype(t̄.a) == Float64
    @test eltype(t̄.b) == Float64
  end

  @testset "Unitful" begin
    d = georef((; z=[1.0, 0.0, 1.0]u"K"))
    poly = GeoStatsModels.fit(Polynomial(), d)
    pred = GeoStatsModels.predict(poly, :z, Point(0.0))
    @test unit(pred) == u"K"
  end

  @testset "CoDa" begin
    d = georef((; z=[Composition(0.1, 0.2), Composition(0.3, 0.4), Composition(0.5, 0.6)]))
    poly = GeoStatsModels.fit(Polynomial(), d)
    pred = GeoStatsModels.predict(poly, :z, Point(0.0))
    @test pred isa Composition
  end
end
