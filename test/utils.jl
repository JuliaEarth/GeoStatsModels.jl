@testset "fitpredict" begin
  rng = StableRNG(2024)
  pset = PointSet(rand(rng, Point, 3))
  gtb = georef((a=[1, 2, 3], b=[4, 5, 6]), pset)
  pred = GeoStatsModels.fitpredict(IDW(), gtb, pset, neighbors=false)
  @test pred.a == gtb.a
  @test pred.b == gtb.b
  @test pred.geometry == gtb.geometry

  # also works with views
  vgtb = view(gtb, 1:3)
  vpred = GeoStatsModels.fitpredict(IDW(), vgtb, pset, neighbors=false)
  @test vpred == pred

  gtb = georef((; z=[1.0, 0.0, 1.0]), [(25.0, 25.0), (50.0, 75.0), (75.0, 50.0)])
  grid = CartesianGrid((100, 100), (0.5, 0.5), (1.0, 1.0))
  linds = LinearIndices(size(grid))
  variogram = GaussianVariogram(range=35.0, nugget=0.0)

  pred = GeoStatsModels.fitpredict(Kriging(variogram), gtb, grid, maxneighbors=3)
  @test isapprox(pred.z[linds[25, 25]], 1.0, atol=1e-3)
  @test isapprox(pred.z[linds[50, 75]], 0.0, atol=1e-3)
  @test isapprox(pred.z[linds[75, 50]], 1.0, atol=1e-3)
end
