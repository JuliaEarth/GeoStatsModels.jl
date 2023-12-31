@testset "NN" begin
  @testset "Basics" begin
    d = georef((; z=["a", "b", "c"]))
    nn = GeoStatsModels.fit(NN(), d)
    pred = GeoStatsModels.predict(nn, :z, Point(0.5))
    @test pred == "a"
    pred = GeoStatsModels.predict(nn, :z, Point(1.5))
    @test pred == "b"
    pred = GeoStatsModels.predict(nn, :z, Point(2.5))
    @test pred == "c"
  end

  @testset "Unitful" begin
    d = georef((; z=[1.0, 0.0, 1.0]u"K"))
    nn = GeoStatsModels.fit(NN(), d)
    pred = GeoStatsModels.predict(nn, :z, Point(0.0))
    @test unit(pred) == u"K"

    # affine units
    d = georef((; z=[1.0, 0.0, 1.0]u"°C"))
    nn = GeoStatsModels.fit(NN(), d)
    #pred = GeoStatsModels.predict(nn, :z, Point(0.0))
    #@test unit(pred) == u"K"
  end

  @testset "CoDa" begin
    d = georef((; z=[Composition(0.1, 0.2), Composition(0.3, 0.4), Composition(0.5, 0.6)]))
    nn = GeoStatsModels.fit(NN(), d)
    pred = GeoStatsModels.predict(nn, :z, Point(0.0))
    @test pred isa Composition
  end
end
