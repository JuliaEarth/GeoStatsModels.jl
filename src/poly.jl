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

struct PolynomialState{D,U}
  coeffs::D
  lenunit::U
end

struct FittedPolynomial{M<:Polynomial,S<:PolynomialState}
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
  x(i) = ustrip.(to(centroid(D, i)))
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

  # length units of coordinates
  lenunit = unit(Meshes.lentype(D))

  # record state
  state = PolynomialState(Dict(vars .=> coeffs), lenunit)

  # return fitted model
  FittedPolynomial(model, state)
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedPolynomial, var, uₒ) = evalpoly(fitted, var, uₒ)

predictprob(fitted::FittedPolynomial, var, uₒ) = Dirac(predict(fitted, var, uₒ))

function evalpoly(fitted::FittedPolynomial, var, uₒ)
  θ = fitted.state.coeffs
  u = fitted.state.lenunit
  d = fitted.model.degree
  xₒ = ustrip.(u, to(centroid(uₒ)))
  V = vandermonde((xₒ,), d)
  first(V * θ[var])
end

function vandermonde(xs, d)
  x = first(xs)
  n = length(x)
  es = Iterators.flatten(multiexponents(n, d) for d in 0:d)
  ps = [[prod(x .^ e) for x in xs] for e in es]
  reduce(hcat, ps)
end
