@testset "LWR" begin
  @testset "Unitful" begin
    d = georef((; z=[1.0, 0.0, 1.0]u"K"))
    lwr = GeoStatsModels.fit(LWR(), d)
    pred = GeoStatsModels.predict(lwr, :z, Point(0.0))
    @test unit(pred) == u"K"

    # affine units
    d = georef((; z=[1.0, 0.0, 1.0]u"°C"))
    lwr = GeoStatsModels.fit(LWR(), d)
    #pred = GeoStatsModels.predict(lwr, :z, Point(0.0))
    #@test unit(pred) == u"K"
  end

  @testset "CoDa" begin
    d = georef((; z=[Composition(0.1, 0.2), Composition(0.3, 0.4), Composition(0.5, 0.6)]))
    lwr = GeoStatsModels.fit(LWR(), d)
    pred = GeoStatsModels.predict(lwr, :z, Point(0.0))
    @test pred isa Composition
  end
end
