# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module GeoStatsModels

using Meshes
using GeoTables
using Variography

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
  Kriging,
  IDW,
  LWR

end
