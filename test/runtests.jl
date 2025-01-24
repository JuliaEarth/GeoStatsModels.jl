using GeoStatsModels
using Meshes
using GeoTables
using CoordRefSystems
using GeoStatsFunctions
using CoDa
using Unitful
using Distances
using LinearAlgebra
using Statistics
using Test, StableRNGs

# list of tests
testfiles = ["nn.jl", "idw.jl", "lwr.jl", "poly.jl", "krig.jl", "misc.jl"]

@testset "GeoStatsModels.jl" begin
  for testfile in testfiles
    include(testfile)
  end
end
