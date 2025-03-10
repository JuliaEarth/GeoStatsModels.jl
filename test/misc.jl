@testset "Miscellaneous" begin
  # fitpredict with views
  pset = PointSet([(25.0, 25.0), (50.0, 75.0), (75.0, 25.0)])
  data = georef((z=[1, 2, 3],), pset)
  model = IDW()
  pred = GeoStatsModels.fitpredict(model, data, pset, neighbors=false)
  @test pred.z == data.z
  @test pred.geometry == data.geometry
  vdata = view(data, 1:3)
  vpred = GeoStatsModels.fitpredict(model, vdata, pset, neighbors=false)
  @test vpred == pred

  # fitpredict with multiple variables and CoKriging
  data = georef((; a=[1.0, 0.0, 0.0], b=[0.0, 1.0, 0.0], c=[0.0, 0.0, 1.0]), [(25.0, 25.0), (50.0, 75.0), (75.0, 50.0)])
  grid = CartesianGrid((100, 100), (0.5, 0.5), (1.0, 1.0))
  model = Kriging([1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0] * SphericalVariogram(range=35.0))
  pred = GeoStatsModels.fitpredict(model, data, grid, maxneighbors=3)
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
  pred = GeoStatsModels.fitpredict(model, data, grid, maxneighbors=3, prob=true)
  @test isapprox(mean(pred.a[inds[25, 25]]), 1.0, atol=1e-3)
  @test isapprox(mean(pred.a[inds[50, 75]]), 0.0, atol=1e-3)
  @test isapprox(mean(pred.a[inds[75, 50]]), 0.0, atol=1e-3)
  @test isapprox(mean(pred.b[inds[25, 25]]), 0.0, atol=1e-3)
  @test isapprox(mean(pred.b[inds[50, 75]]), 1.0, atol=1e-3)
  @test isapprox(mean(pred.b[inds[75, 50]]), 0.0, atol=1e-3)
  @test isapprox(mean(pred.c[inds[25, 25]]), 0.0, atol=1e-3)
  @test isapprox(mean(pred.c[inds[50, 75]]), 0.0, atol=1e-3)
  @test isapprox(mean(pred.c[inds[75, 50]]), 1.0, atol=1e-3)

  # fitpredict with neighbors
  data = georef((; z=[1.0, 0.0, 1.0]), [(25.0, 25.0), (50.0, 75.0), (75.0, 50.0)])
  grid = CartesianGrid(100, 100)
  models = [NN(), IDW(), LWR(), Polynomial(), Kriging(SphericalVariogram())]
  for model in models
    pred = GeoStatsModels.fitpredict(model, data, grid, maxneighbors=3)
    @test pred.geometry == grid
  end
end
