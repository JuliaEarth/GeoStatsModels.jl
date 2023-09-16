@testset "IDW" begin
  @testset "Unitful" begin
    d = georef((; z=[1.0, 0.0, 1.0]u"K"))
    idw = GeoStatsModels.fit(IDW(), d)
    pred = GeoStatsModels.predict(idw, :z, Point(0.0))
    @test unit(pred) == u"K"

    # affine units
    d = georef((; z=[1.0, 0.0, 1.0]u"°C"))
    idw = GeoStatsModels.fit(IDW(), d)
    #pred = GeoStatsModels.predict(idw, :z, Point(0.0))
    #@test unit(pred) == u"K"
  end

  @testset "CoDa" begin
    d = georef((; z=[Composition(0.1, 0.2), Composition(0.3, 0.4), Composition(0.5, 0.6)]))
    idw = GeoStatsModels.fit(IDW(), d)
    pred = GeoStatsModels.predict(idw, :z, Point(0.0))
    @test pred isa Composition
  end
end