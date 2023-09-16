# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    LWR(weightfun=h -> exp(-3 * h^2), distance=Euclidean())

The locally weighted regression (a.k.a. LOESS) model introduced by
Cleveland 1979. It is the most natural generalization of [`IDW`](@ref)
in which one is allowed to use a custom weight function instead of
distance-based weights.

## References

* Stone 1977. [Consistent non-parametric regression](https://tinyurl.com/4da68xxf)
* Cleveland 1979. [Robust locally weighted regression and smoothing
  scatterplots](https://www.tandfonline.com/doi/abs/10.1080/01621459.1979.10481038)
* Cleveland & Grosse 1991. [Computational methods for local
  regression](https://link.springer.com/article/10.1007/BF01890836)
"""
struct LWR{F,D} <: GeoStatsModel
  weightfun::F
  distance::D
end

LWR(weightfun) = LWR(weightfun, Euclidean())
LWR() = LWR(h -> exp(-3 * h ^ 2))

struct LWRState{D<:AbstractGeoTable}
  data::D
end

struct FittedLWR{M<:LWR,S<:LWRState}
  model::M
  state::S
end

status(fitted::FittedLWR) = true

#--------------
# FITTING STEP
#--------------

function fit(model::LWR, data)
  # record state
  state = LWRState(data)

  # return fitted model
  FittedLWR(model, state)
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedLWR, var, uₒ) = first(lwr(fitted, var, uₒ))

function predictprob(fitted::FittedLWR, var, uₒ)
  μ, σ² = lwr(fitted, var, uₒ)
  Normal(μ, √σ²)
end

function lwr(fitted::FittedLWR, var, uₒ)
  w = fitted.model.weightfun
  δ = fitted.model.distance
  d = fitted.state.data
  c = Tables.columns(values(d))
  z = Tables.getcolumn(c, var)

  Ω = domain(d)
  n = nelements(Ω)

  x(i) = coordinates(centroid(Ω, i))

  # fit step
  X = mapreduce(x, hcat, 1:n)
  Xₗ = [ones(eltype(X), n) X']
  zₗ = z

  # predict step
  xₒ = coordinates(uₒ)
  δs = map(i -> δ(xₒ, x(i)), 1:n)
  δs .= δs ./ maximum(δs)
  Wₗ = Diagonal(w.(δs))
  θₗ = Xₗ' * Wₗ * Xₗ \ Xₗ' * Wₗ * zₗ
  xₗ = [one(eltype(xₒ)); xₒ]
  rₗ = Wₗ * Xₗ * (Xₗ' * Wₗ * Xₗ \ xₗ)

  μ = θₗ ⋅ xₗ
  σ² = norm(rₗ)

  μ, σ²
end
