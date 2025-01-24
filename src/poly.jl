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

struct PolynomialState{D<:AbstractGeoTable,C}
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
  d = model.degree
  D = domain(data)

  # multivariate Vandermonde matrix
  x(i) = CoordRefSystems.raw(coords(centroid(D, i)))
  xs = (x(i) for i in 1:nelements(D))
  V = vandermonde(xs, d)

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
  D = domain(fitted.state.data)
  θ = fitted.state.coeffs
  d = fitted.model.degree
  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(D))
  xₒ = CoordRefSystems.raw(coords(centroid(gₒ′)))
  V = vandermonde((xₒ,), d)
  first(V * θ[var])
end

function vandermonde(xs, d)
  n = length(first(xs))
  I = (multiexponents(n, d) for d in 0:d)
  es = Iterators.flatten(I) |> collect
  [prod(x .^ e) for x in xs, e in es]
end
