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

struct FittedLWR{M<:LWR,S<:LWRState} <: FittedGeoStatsModel
  model::M
  state::S
end

status(fitted::FittedLWR) = true

#--------------
# FITTING STEP
#--------------

function fit(model::LWR, data)
  # preallocate coordinate matrix
  X = prealloc(model, data)

  # set coordinate matrix
  setx!(model, X, data)

  # record state
  state = LWRState(data, X)

  # return fitted model
  FittedLWR(model, state)
end

function prealloc(::LWR, data)
  dom = domain(data)
  nobs = nelements(dom)
  x = CoordRefSystems.raw(coords(centroid(dom, 1)))
  Matrix{eltype(x)}(undef, nobs, length(x) + 1)
end

function setx!(::LWR, X, data)
  dom = domain(data)
  nobs = nelements(dom)
  x(i) = CoordRefSystems.raw(coords(centroid(dom, i)))
  @inbounds for i in 1:nobs
    X[i, 1] = oneunit(eltype(X))
    X[i, 2:end] .= x(i)
  end
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedLWR, var::Symbol, gₒ) = predictmean(fitted, var, gₒ)

function predictprob(fitted::FittedLWR, var::Symbol, gₒ)
  X, W, A, x, z = matrices(fitted, var, gₒ)
  μ = lwrmean(X, W, A, x, z)
  σ² = lwrvar(X, W, A, x)
  Normal(μ, √σ²)
end

function predictmean(fitted::FittedLWR, var, gₒ)
  X, W, A, x, z = matrices(fitted, var, gₒ)
  lwrmean(X, W, A, x, z)
end

function matrices(fitted::FittedLWR, var, gₒ)
  d = fitted.state.data
  c = Tables.columns(values(d))
  z = Tables.getcolumn(c, var)

  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(domain(d)))

  X = fitted.state.X
  W = wmatrix(fitted, gₒ′)
  A = transpose(X) * W * X

  xₒ = CoordRefSystems.raw(coords(centroid(gₒ′)))
  x = [one(eltype(xₒ)), xₒ...]

  X, W, A, x, z
end

function wmatrix(fitted::FittedLWR, gₒ)
  w = fitted.model.weightfun
  δ = fitted.model.distance
  d = fitted.state.data
  Ω = domain(d)
  n = nelements(Ω)

  pₒ = centroid(gₒ)
  p(i) = centroid(Ω, i)

  δs = map(i -> evaluate(δ, pₒ, p(i)), 1:n)
  ws = w.(δs / maximum(δs))

  Diagonal(ws)
end

function lwrmean(X, W, A, x, z)
  θ = A \ X' * (W * z)
  sum(i -> x[i] * θ[i], eachindex(x, θ))
end

function lwrvar(X, W, A, x)
  r = W * X * (A \ x)
  norm(r)
end
