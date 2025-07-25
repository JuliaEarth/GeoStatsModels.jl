# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    IDW(exponent=1, distance=Euclidean())

The inverse distance weighting model introduced in
the very early days of geostatistics by Shepard 1968.

The weights are computed as `λᵢ = 1 / d(x, xᵢ)ᵉ` for
a given `distance` denoted by `d` and `exponent` denoted
by `e`.

## References

* Shepard 1968. [A two-dimensional interpolation function
  for irregularly-spaced data](https://dl.acm.org/doi/10.1145/800186.810616)
"""
struct IDW{E,D} <: GeoStatsModel
  exponent::E
  distance::D
end

IDW(exponent) = IDW(exponent, Euclidean())
IDW() = IDW(1)

struct IDWState{D<:AbstractGeoTable}
  data::D
end

struct FittedIDW{M<:IDW,S<:IDWState} <: FittedGeoStatsModel
  model::M
  state::S
end

status(fitted::FittedIDW) = true

#--------------
# FITTING STEP
#--------------

function fit(model::IDW, data)
  # record state
  state = IDWState(data)

  # return fitted model
  FittedIDW(model, state)
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedIDW, var::Symbol, gₒ) = idw(fitted, weights(fitted, gₒ), var)

predictprob(fitted::FittedIDW, var::Symbol, gₒ) = Dirac(predict(fitted, var, gₒ))

function idw(fitted::FittedIDW, weights, var)
  d = fitted.state.data
  c = Tables.columns(values(d))
  z = Tables.getcolumn(c, var)
  w = weights
  Σw = sum(w)

  λ(i) = w[i] / Σw

  if isinf(Σw) # some distance is zero?
    z[findfirst(isinf, w)]
  else
    sum(i -> λ(i) * z[i], eachindex(z))
  end
end

function weights(fitted::FittedIDW, gₒ)
  e = fitted.model.exponent
  δ = fitted.model.distance
  d = fitted.state.data
  Ω = domain(d)

  # obtain centroid and adjust CRS
  pₒ = centroid(gₒ) |> Proj(crs(Ω))

  p(i) = centroid(Ω, i)

  λ(i) = 1 / evaluate(δ, pₒ, p(i)) ^ e

  map(λ, 1:nelements(Ω))
end
