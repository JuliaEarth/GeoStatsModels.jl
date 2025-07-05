@testset "Utilities" begin
  if Threads.nthreads() > 1
    @test GeoStatsModels.isthreaded()
  end
  @test !GeoStatsModels.isthreaded(false)
end
