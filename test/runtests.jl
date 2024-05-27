using GeoStatsModels
using Meshes
using GeoTables
using GeoStatsFunctions
using CoDa
using Unitful
using LinearAlgebra
using Statistics
using Test, StableRNGs

# list of tests
testfiles = ["krig.jl", "nn.jl", "idw.jl", "lwr.jl", "utils.jl"]

@testset "GeoStatsModels.jl" begin
  for testfile in testfiles
    include(testfile)
  end
end
