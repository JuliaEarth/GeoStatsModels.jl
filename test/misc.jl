@testset "Miscellaneous" begin
  # fitpredict with views
  model = IDW()
  data = georef((z=[1, 2, 3],), [(25.0, 25.0), (50.0, 75.0), (75.0, 25.0)])
  pset = domain(data)
  pred = GeoStatsModels.fitpredict(model, data, pset, neighbors=false)
  @test pred.z == data.z
  @test pred.geometry == data.geometry
  vdata = view(data, 1:3)
  vpred = GeoStatsModels.fitpredict(model, vdata, pset, neighbors=false)
  @test vpred == pred

  # fitpredict with Kriging (without neighbors)
  model = Kriging(SphericalVariogram(range=35.0))
  data = georef((; z=[1.0, 0.0, 1.0]), [(25.0, 25.0), (50.0, 75.0), (75.0, 50.0)])
  grid = CartesianGrid((100, 100), (0.5, 0.5), (1.0, 1.0))
  inds = LinearIndices(size(grid))
  pred = GeoStatsModels.fitpredict(model, data, grid, neighbors=false)
  @test isapprox(pred.z[inds[25, 25]], 1.0, atol=1e-3)
  @test isapprox(pred.z[inds[50, 75]], 0.0, atol=1e-3)
  @test isapprox(pred.z[inds[75, 50]], 1.0, atol=1e-3)
  pred = GeoStatsModels.fitpredict(model, data, grid, neighbors=true)
  @test isapprox(pred.z[inds[25, 25]], 1.0, atol=1e-3)
  @test isapprox(pred.z[inds[50, 75]], 0.0, atol=1e-3)
  @test isapprox(pred.z[inds[75, 50]], 1.0, atol=1e-3)

  # fitpredict with multiple variables and CoKriging
  model = Kriging([1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0] * SphericalVariogram(range=35.0))
  data = georef((; a=[1.0, 0.0, 0.0], b=[0.0, 1.0, 0.0], c=[0.0, 0.0, 1.0]), [(25.0, 25.0), (50.0, 75.0), (75.0, 50.0)])
  grid = CartesianGrid((100, 100), (0.5, 0.5), (1.0, 1.0))
  inds = LinearIndices(size(grid))
  pred = GeoStatsModels.fitpredict(model, data, grid, neighbors=true)
  @test isapprox(pred.a[inds[25, 25]], 1.0, atol=1e-3)
  @test isapprox(pred.a[inds[50, 75]], 0.0, atol=1e-3)
  @test isapprox(pred.a[inds[75, 50]], 0.0, atol=1e-3)
  @test isapprox(pred.b[inds[25, 25]], 0.0, atol=1e-3)
  @test isapprox(pred.b[inds[50, 75]], 1.0, atol=1e-3)
  @test isapprox(pred.b[inds[75, 50]], 0.0, atol=1e-3)
  @test isapprox(pred.c[inds[25, 25]], 0.0, atol=1e-3)
  @test isapprox(pred.c[inds[50, 75]], 0.0, atol=1e-3)
  @test isapprox(pred.c[inds[75, 50]], 1.0, atol=1e-3)
  pred = GeoStatsModels.fitpredict(model, data, grid, neighbors=true, prob=true)
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
  for model in [NN(), IDW(), LWR(), Polynomial(), Kriging(SphericalVariogram())]
    pred = GeoStatsModels.fitpredict(model, data, grid, neighbors=true)
    @test pred.geometry == grid
  end
end
