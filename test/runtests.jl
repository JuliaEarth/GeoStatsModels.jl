using GeoStatsModels
using Meshes
using GeoTables
using Variography
using CoDa
using Unitful
using LinearAlgebra
using Statistics
using Test, Random

# list of tests
testfiles = ["kriging.jl"]

@testset "KrigingEstimators.jl" begin
  for testfile in testfiles
    include(testfile)
  end
end
