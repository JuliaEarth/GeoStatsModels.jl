@testset "Miscellaneous" begin
  rng = StableRNG(2024)

  # fitpredict with IDW
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

  # fitpredict with multiple variables and CoKriging
  gtb = georef((; a=[1.0, 0.0, 0.0], b=[0.0, 1.0, 0.0], c=[0.0, 0.0, 1.0]), [(25.0, 25.0), (50.0, 75.0), (75.0, 50.0)])
  grid = CartesianGrid((100, 100), (0.5, 0.5), (1.0, 1.0))
  model = Kriging([1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0] * SphericalVariogram(range=35.0))
  pred = GeoStatsModels.fitpredict(model, gtb, grid, maxneighbors=3)
  inds = LinearIndices(size(grid))
  @test isapprox(pred.a[inds[25, 25]], 1.0, atol=1e-3)
  @test isapprox(pred.a[inds[50, 75]], 0.0, atol=1e-3)
  @test isapprox(pred.a[inds[75, 50]], 0.0, atol=1e-3)
  @test isapprox(pred.b[inds[25, 25]], 0.0, atol=1e-3)
  @test isapprox(pred.b[inds[50, 75]], 1.0, atol=1e-3)
  @test isapprox(pred.b[inds[75, 50]], 0.0, atol=1e-3)
  @test isapprox(pred.c[inds[25, 25]], 0.0, atol=1e-3)
  @test isapprox(pred.c[inds[50, 75]], 0.0, atol=1e-3)
  @test isapprox(pred.c[inds[75, 50]], 1.0, atol=1e-3)
end
