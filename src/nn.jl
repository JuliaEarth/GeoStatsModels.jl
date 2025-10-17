# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    NN(distance=Euclidean())

A model that assigns the nearest non-missing value from neighbors.
"""
struct NN{D} <: GeoStatsModel
  distance::D
end

NN() = NN(Euclidean())

struct NNState{D<:AbstractGeoTable}
  data::D
end

struct FittedNN{M<:NN,S<:NNState} <: FittedGeoStatsModel
  model::M
  state::S
end

status(fitted::FittedNN) = true

#--------------
# FITTING STEP
#--------------

function fit(model::NN, data)
  # record state
  state = NNState(data)

  # return fitted model
  FittedNN(model, state)
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedNN, var::Symbol, gₒ) = nn(fitted, distances(fitted, gₒ), var)

predictprob(fitted::FittedNN, var::Symbol, gₒ) = Dirac(predict(fitted, var, gₒ))

function nn(fitted::FittedNN, distances, var)
  d = fitted.state.data
  c = Tables.columns(values(d))
  z = Tables.getcolumn(c, var)
  s = z[sortperm(distances)]
  i = findfirst(!ismissing, s)
  isnothing(i) ? missing : s[i]
end

function distances(fitted::FittedNN, gₒ)
  δ = fitted.model.distance
  d = fitted.state.data
  Ω = domain(d)

  # obtain centroid and adjust CRS
  pₒ = centroid(gₒ) |> Proj(crs(Ω))

  p(i) = centroid(Ω, i)

  λ(i) = evaluate(δ, pₒ, p(i))

  map(λ, 1:nelements(Ω))
end
