# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module GeoStatsModels

using Meshes
using GeoTables
using GeoStatsFunctions

using LinearAlgebra
using Distributions
using Combinatorics
using Distances
using Unitful
using Tables

using Unitful: AffineQuantity

include("models.jl")

# utility functions
include("utils.jl")

export
  # models
  NN,
  IDW,
  LWR,
  Kriging

end
