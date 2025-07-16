# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    Polynomial(degree=1)

A polynomial model with coefficients obtained via regression.
"""
struct Polynomial <: GeoStatsModel
  degree::Int
end

Polynomial() = Polynomial(1)

struct PolynomialState{D<:AbstractGeoTable,P}
  data::D
  proj::P
end

struct FittedPolynomial{M<:Polynomial,S<:PolynomialState} <: FittedGeoStatsModel
  model::M
  state::S
end

status(fitted::FittedPolynomial) = true

#--------------
# FITTING STEP
#--------------

function fit(model::Polynomial, data)
  # preallocate regression matrix
  proj = prealloc(model, data)

  # set regression matrix
  setproj!(model, proj, data)

  # record state
  state = PolynomialState(data, proj)

  # return fitted model
  FittedPolynomial(model, state)
end

function prealloc(model::Polynomial, data)
  # retrieve parameters
  deg = model.degree
  dom = domain(data)

  # raw coordinates of centroid
  x = CoordRefSystems.raw(coords(centroid(dom, 1)))
  n = length(x)

  # size of regression matrix
  iter = (multiexponents(n, d) for d in 0:deg)
  nexp = sum(length, iter)
  nobs = nelements(dom)

  Matrix{eltype(x)}(undef, nexp, nobs)
end

function setproj!(model::Polynomial, proj, data)
  # retrieve parameters
  deg = model.degree
  dom = domain(data)

  # raw coordinates of centroids
  x(i) = CoordRefSystems.raw(coords(centroid(dom, i)))
  xs = (x(i) for i in 1:nelements(dom))

  # multivariate Vandermonde matrix
  V = vandermonde(xs, deg)

  # set regression matrix
  proj .= (transpose(V) * V) \ transpose(V)

  nothing
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedPolynomial, var::Symbol, gₒ) = evalpoly(fitted, var, gₒ)

predictprob(fitted::FittedPolynomial, var::Symbol, gₒ) = Dirac(predict(fitted, var, gₒ))

function evalpoly(fitted::FittedPolynomial, var, gₒ)
  # retrieve degree and data
  deg = fitted.model.degree
  data = fitted.state.data

  # obtain centroid and adjust CRS
  pₒ = centroid(gₒ) |> Proj(crs(domain(data)))

  # raw coordinates of centroid
  xₒ = CoordRefSystems.raw(coords(pₒ))

  # multivariate Vandermonde matrix
  V = vandermonde((xₒ,), deg)

  # regression coefficients
  P = fitted.state.proj
  c = Tables.columns(values(data))
  z = Tables.getcolumn(c, var)
  θ = P * z

  first(V * θ)
end

function vandermonde(xs, deg)
  n = length(first(xs))
  iter = (multiexponents(n, d) for d in 0:deg)
  exps = Iterators.flatten(iter) |> collect
  [prod(x .^ e) for x in xs, e in exps]
end
