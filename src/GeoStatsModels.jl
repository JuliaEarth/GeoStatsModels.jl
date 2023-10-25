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

import StatsAPI: fit, predict

include("models.jl")
include("krig.jl")
include("idw.jl")
include("lwr.jl")

# utility functions
include("fitpredict.jl")

export
  # models
  Kriging,
  IDW,
  LWR,
  fit,
  status,
  predict,
  predictprob

end
