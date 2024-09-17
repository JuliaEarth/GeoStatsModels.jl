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

