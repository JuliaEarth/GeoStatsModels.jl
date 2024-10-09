# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    NN(distance=Euclidean())

A model that assigns the value of the nearest observation.
"""
struct NN{D} <: GeoStatsModel
  distance::D
end

NN() = NN(Euclidean())

struct NNState{D<:AbstractGeoTable}
  data::D
end

struct FittedNN{M<:NN,S<:NNState}
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

predict(fitted::FittedNN, var, uₒ) = nn(fitted, distances(fitted, uₒ), var)

predictprob(fitted::FittedNN, var, uₒ) = Dirac(predict(fitted, var, uₒ))

function nn(fitted::FittedNN, distances, var)
  d = fitted.state.data
  c = Tables.columns(values(d))
  z = Tables.getcolumn(c, var)
  z[argmin(distances)]
end

function distances(fitted::FittedNN, uₒ)
  δ = fitted.model.distance
  d = fitted.state.data
  Ω = domain(d)

  pₒ = centroid(uₒ)
  p(i) = centroid(Ω, i)

  λ(i) = evaluate(δ, pₒ, p(i))

  map(λ, 1:nelements(Ω))
end
