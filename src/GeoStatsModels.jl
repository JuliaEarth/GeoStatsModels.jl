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
using Unitful
using Tables

include("models.jl")
include("krig.jl")
include("idw.jl")
include("lwr.jl")

export
  # models
  SimpleKriging,
  OrdinaryKriging,
  UniversalKriging,
  ExternalDriftKriging,
  IDW,
  LWR

end
