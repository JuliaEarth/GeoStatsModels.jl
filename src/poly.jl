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

mutable struct PolynomialState{D<:AbstractGeoTable,C}
  data::D
  coeffs::C
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

  # multivariate Vandermonde matrix
  x(i) = CoordRefSystems.raw(coords(centroid(dom, i)))
  xs = (x(i) for i in 1:nelements(dom))
  V = vandermonde(xs, deg)

  # regression matrix
  P = V'V \ V'

  # regression coefficients
  cols = Tables.columns(values(data))
  vars = Tables.columnnames(cols)
  coeffs = map(vars) do var
    P * Tables.getcolumn(cols, var)
  end

  # record state
  state = PolynomialState(data, Dict(vars .=> coeffs))

  # return fitted model
  FittedPolynomial(model, state)
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedPolynomial, var::Symbol, gₒ) = evalpoly(fitted, var, gₒ)

predictprob(fitted::FittedPolynomial, var::Symbol, gₒ) = Dirac(predict(fitted, var, gₒ))

function evalpoly(fitted::FittedPolynomial, var, gₒ)
  θ = fitted.state.coeffs
  deg = fitted.model.degree
  dom = domain(fitted.state.data)
  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(dom))
  xₒ = CoordRefSystems.raw(coords(centroid(gₒ′)))
  V = vandermonde((xₒ,), deg)
  first(V * θ[var])
end

function vandermonde(xs, deg)
  n = length(first(xs))
  I = (multiexponents(n, d) for d in 0:deg)
  es = Iterators.flatten(I) |> collect
  [prod(x .^ e) for x in xs, e in es]
end
