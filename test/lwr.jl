@testset "LWR" begin
  @testset "Unitful" begin
    data = georef((; z=[1.0, 0.0, 1.0]u"K"))
    lwr = GeoStatsModels.fit(LWR(), data)
    pred = GeoStatsModels.predict(lwr, :z, Point(0.0))
    @test unit(pred) == u"K"

    # latlon coordinates
    data = georef((; z=[1.0, 0.0, 1.0]), Point.([LatLon(0, 0), LatLon(0, 1), LatLon(1, 0)]))
    lwr = GeoStatsModels.fit(LWR(h -> exp(-3 * h^2), Haversine()), data)
    pred = GeoStatsModels.predict(lwr, :z, Point(LatLon(0, 0)))
    @test pred ≈ 1
    pred = GeoStatsModels.predict(lwr, :z, Point(LatLon(0, 1)))
    @test pred ≈ 0
    pred = GeoStatsModels.predict(lwr, :z, Point(LatLon(1, 0)))
    @test pred ≈ 1
  end

  @testset "CoDa" begin
    data = georef((; z=[Composition(0.1, 0.2), Composition(0.3, 0.4), Composition(0.5, 0.6)]))
    lwr = GeoStatsModels.fit(LWR(), data)
    pred = GeoStatsModels.predict(lwr, :z, Point(0.0))
    @test pred isa Composition
  end

  @testset "Fallbacks" begin
    data = georef((; z=[1.0, 0.0, 1.0]), [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    lwr = GeoStatsModels.fit(LWR(), data)
    pred1 = GeoStatsModels.predict(lwr, :z, Point(0.0, 0.0))
    pred2 = GeoStatsModels.predict(lwr, "z", Point(0.0, 0.0))
    pred3 = GeoStatsModels.predict(lwr, (:z,), Point(0.0, 0.0))
    pred4 = GeoStatsModels.predictprob(lwr, :z, Point(0.0, 0.0))
    pred5 = GeoStatsModels.predictprob(lwr, "z", Point(0.0, 0.0))
    pred6 = GeoStatsModels.predictprob(lwr, (:z,), Point(0.0, 0.0))
    @test pred1 isa Number
    @test pred2 isa Number
    @test pred3 isa AbstractVector
    @test pred4 isa Normal
    @test pred5 isa Normal
    @test pred6 isa DiagNormal
    @test pred1 == pred2
    @test pred1 == pred3[1]
  end
end
