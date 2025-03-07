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
  # retrieve parameters
  deg = model.degree
  dom = domain(data)

  # raw coordinates of centroids
  x(i) = CoordRefSystems.raw(coords(centroid(dom, i)))
  xs = (x(i) for i in 1:nelements(dom))

  # multivariate Vandermonde matrix
  V = vandermonde(xs, deg)

  # regression matrix
  proj = (transpose(V) * V) \ transpose(V)

  # record state
  state = PolynomialState(data, proj)

  # return fitted model
  FittedPolynomial(model, state)
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedPolynomial, var::Symbol, gₒ) = evalpoly(fitted, var, gₒ)

predictprob(fitted::FittedPolynomial, var::Symbol, gₒ) = Dirac(predict(fitted, var, gₒ))

function evalpoly(fitted::FittedPolynomial, var, gₒ)
  deg = fitted.model.degree
  data = fitted.state.data
  proj = fitted.state.proj

  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(domain(data)))

  # raw coordinates of centroid
  xₒ = CoordRefSystems.raw(coords(centroid(gₒ′)))

  # multivariate Vandermonde matrix
  V = vandermonde((xₒ,), deg)

  # regression coefficients
  c = Tables.columns(values(data))
  z = Tables.getcolumn(c, var)
  θ = proj * z

  first(V * θ)
end

function vandermonde(xs, deg)
  n = length(first(xs))
  I = (multiexponents(n, d) for d in 0:deg)
  es = Iterators.flatten(I) |> collect
  [prod(x .^ e) for x in xs, e in es]
end
