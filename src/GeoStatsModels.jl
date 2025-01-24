# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module GeoStatsModels

using Meshes
using GeoTables
using CoordRefSystems
using GeoStatsFunctions

using LinearAlgebra
using Distributions
using Combinatorics
using Distances
using Unitful
using Tables

include("models.jl")

export
  # models
  NN,
  IDW,
  LWR,
  Polynomial,
  Kriging

end
