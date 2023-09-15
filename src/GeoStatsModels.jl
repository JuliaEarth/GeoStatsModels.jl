# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module GeoStatsModels

using Meshes
using GeoTables
using Variography

using LinearAlgebra: Factorization, Symmetric
using LinearAlgebra: bunchkaufman, cholesky
using LinearAlgebra: issuccess, ⋅
using Combinatorics: multiexponents
using Distributions: Normal, Dirac
using Unitful
using Tables

include("models.jl")
include("kriging.jl")
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
