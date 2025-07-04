@testset "NN" begin
  @testset "Basics" begin
    data = georef((; z=["a", "b", "c"]))
    nn = GeoStatsModels.fit(NN(), data)
    pred = GeoStatsModels.predict(nn, :z, Point(0.5))
    @test pred == "a"
    pred = GeoStatsModels.predict(nn, :z, Point(1.5))
    @test pred == "b"
    pred = GeoStatsModels.predict(nn, :z, Point(2.5))
    @test pred == "c"

    # latlon coordinates
    data = georef((; z=["a", "b", "c"]), Point.([LatLon(0, 1), LatLon(0, 2), LatLon(0, 3)]))
    nn = GeoStatsModels.fit(NN(Haversine()), data)
    pred = GeoStatsModels.predict(nn, :z, Point(LatLon(0, 0.8)))
    @test pred == "a"
    pred = GeoStatsModels.predict(nn, :z, Point(LatLon(0, 1.8)))
    @test pred == "b"
    pred = GeoStatsModels.predict(nn, :z, Point(LatLon(0, 2.8)))
    @test pred == "c"
  end

  @testset "Unitful" begin
    data = georef((; z=[1.0, 0.0, 1.0]u"K"))
    nn = GeoStatsModels.fit(NN(), data)
    pred = GeoStatsModels.predict(nn, :z, Point(0.0))
    @test unit(pred) == u"K"
  end

  @testset "CoDa" begin
    data = georef((; z=[Composition(0.1, 0.2), Composition(0.3, 0.4), Composition(0.5, 0.6)]))
    nn = GeoStatsModels.fit(NN(), data)
    pred = GeoStatsModels.predict(nn, :z, Point(0.0))
    @test pred isa Composition
  end

  @testset "Fallbacks" begin
    data = georef((; z=[1.0, 0.0, 1.0]), [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    nn = GeoStatsModels.fit(NN(), data)
    pred1 = GeoStatsModels.predict(nn, :z, Point(0.0, 0.0))
    pred2 = GeoStatsModels.predict(nn, "z", Point(0.0, 0.0))
    pred3 = GeoStatsModels.predict(nn, (:z,), Point(0.0, 0.0))
    pred4 = GeoStatsModels.predictprob(nn, :z, Point(0.0, 0.0))
    pred5 = GeoStatsModels.predictprob(nn, "z", Point(0.0, 0.0))
    pred6 = GeoStatsModels.predictprob(nn, (:z,), Point(0.0, 0.0))
    @test pred1 isa Number
    @test pred2 isa Number
    @test pred3 isa AbstractVector
    @test pred4 isa Dirac
    @test pred5 isa Dirac
    @test pred6 isa Product
    @test pred1 == pred2
    @test pred1 == pred3[1]
  end
end
