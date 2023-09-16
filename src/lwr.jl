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

mutable struct LWRState{D<:AbstractGeoTable,T}
  data::D
  X::T
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
  Ω = domain(data)
  n = nelements(Ω)

  x(i) = coordinates(centroid(Ω, i))

  # coordinates matrix
  X = mapreduce(x, hcat, 1:n)
  X = [ones(eltype(X), n) X']

  # record state
  state = LWRState(data, X)

  # return fitted model
  FittedLWR(model, state)
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedLWR, var, uₒ) = predictmean(fitted, var, uₒ)

function predictprob(fitted::FittedLWR, var, uₒ)
  X, W, A, x, z = matrices(fitted, var, uₒ)
  μ = lwrmean(X, W, A, x, z)
  σ² = lwrvar(X, W, A, x)
  Normal(μ, √σ²)
end

function predictmean(fitted::FittedLWR, var, uₒ)
  X, W, A, x, z = matrices(fitted, var, uₒ)
  lwrmean(X, W, A, x, z)
end

function matrices(fitted::FittedLWR, var, uₒ)
  d = fitted.state.data
  c = Tables.columns(values(d))
  z = Tables.getcolumn(c, var)

  X = fitted.state.X
  W = wmatrix(fitted, uₒ)
  A = X' * W * X

  xₒ = coordinates(uₒ)
  x = [one(eltype(xₒ)); xₒ]

  X, W, A, x, z
end

function wmatrix(fitted::FittedLWR, uₒ)
  w = fitted.model.weightfun
  δ = fitted.model.distance
  d = fitted.state.data
  Ω = domain(d)
  n = nelements(Ω)

  xₒ = coordinates(uₒ)
  x(i) = coordinates(centroid(Ω, i))

  δs = map(i -> δ(xₒ, x(i)), 1:n)
  ws = w.(δs / maximum(δs))

  Diagonal(ws)
end

function lwrmean(X, W, A, x, z)
  θ = A \ X' * (W * z)
  sum(x .* θ)
end

function lwrvar(X, W, A, x)
  r = W * X * (A \ x)
  norm(r)
end
