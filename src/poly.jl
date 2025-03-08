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

mutable struct PolynomialState{D<:AbstractGeoTable,P}
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

function fit!(fitted::FittedPolynomial, newdata)
  model = fitted.model
  state = fitted.state

  # check compatibility of data size
  checkdatasize(fitted, newdata)

  # update state data
  state.data = newdata

  # set coordinate matrix
  setproj!(model, state.proj, newdata)

  nothing
end

function checkdatasize(fitted::FittedPolynomial, data)
  proj = fitted.state.proj
  nproj = size(proj, 2)
  nobs = nrow(data)
  if nobs > nproj
    throw(ArgumentError("in-place fit called with $nobs data row(s) and $nproj maximum size"))
  end
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
  deg = fitted.model.degree
  data = fitted.state.data

  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(domain(data)))

  # raw coordinates of centroid
  xₒ = CoordRefSystems.raw(coords(centroid(gₒ′)))

  # multivariate Vandermonde matrix
  V = vandermonde((xₒ,), deg)

  # regression coefficients
  P = @view fitted.state.proj[:, 1:nrow(data)]
  c = Tables.columns(values(data))
  z = Tables.getcolumn(c, var)
  θ = P * z

  first(V * θ)
end

function vandermonde(xs, deg)
  n = length(first(xs))
  I = (multiexponents(n, d) for d in 0:deg)
  es = Iterators.flatten(I) |> collect
  [prod(x .^ e) for x in xs, e in es]
end
