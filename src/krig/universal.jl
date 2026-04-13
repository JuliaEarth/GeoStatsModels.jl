# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    UniversalKriging(fun, drifts)

Universal Kriging with geostatistical function `fun` and `drifts`.
A drift is a function `p -> v` that maps a point `p` to a scalar
value `v`.

    UniversalKriging(fun, deg, dim)

Alternatively, construct monomial `drifts` up to given degree `deg`
for `dim` geospatial coordinates. For example, if the data is mapped
with `(x, y)` `Cartesian` coordinates, then `dim=2` and setting `deg=1`
will add the monomials `1`, `x`, and `y` as drift functions to the mean
`1 + β₁x + β₂y`, while setting `deg=2` will lead to a quadratic mean
`1 + β₁x + β₂y + β₃x² + β₄xy + β₅y²`. The same logic applies to `(ϕ, λ)`
`LatLon` coordinates or any other type of geospatial coordinates.

## Examples

```julia
# univariate model with mean 1 + β₁x + β₂y where x and y are Cartesian coordinates
UniversalKriging(SphericalVariogram(), [p -> 1, p -> coords(p).x, p -> coords(p).y])

# multivariate model with mean 1 + βx² where x is the first Cartesian coordinate
UniversalKriging(I(2) * SphericalVariogram(), [p -> 1, p -> coords(p).x^2])
```

See also [`SimpleKriging`](@ref) and [`OrdinaryKriging`](@ref)
for related Kriging models and the general [`Kriging`](@ref)
constructor that selects the appropriate variant as a function
of the arguments.

### Notes

Drift functions should be smooth for numerical stability.

Include a constant drift (e.g. `p -> 1`) for unbiased estimation.

[`OrdinaryKriging`](@ref) is recovered with `drifts = [p -> 1]`.
"""
struct UniversalKriging{F<:GeoStatsFunction,D} <: KrigingModel
  fun::F
  drifts::D
end

UniversalKriging(fun::GeoStatsFunction, deg::Int, dim::Int) = UniversalKriging(fun, monomials(deg, dim))

scale(model::UniversalKriging, α) = UniversalKriging(GeoStatsFunctions.scale(model.fun, α), model.drifts)

function monomials(deg::Int, dim::Int)
  # sanity checks
  @assert deg ≥ 0 "degree must be nonnegative"
  @assert dim > 0 "dimension must be positive"

  # helper function to extract raw coordinates
  x(p) = CoordRefSystems.raw(coords(p))

  # build drift functions for given degree and dimension
  map(exponents(deg, dim)) do n
    p -> prod(x(p) .^ n)
  end
end

function exponents(deg::Int, dim::Int)
  # multinomial expansion up to given degree
  pow = reduce(hcat, stack(multiexponents(dim, d)) for d in 0:deg)

  # sort for better conditioned Kriging matrices
  inds = sortperm(vec(maximum(pow, dims=1)), rev=true)

  # return iterator of monomial exponents
  eachcol(pow[:, inds])
end

nconstraints(model::UniversalKriging) = nvariates(model.fun) * length(model.drifts)

function lhsconstraints!(model::UniversalKriging, LHS::AbstractMatrix, domain)
  drifts = model.drifts
  nobs = nelements(domain)
  nvar = nvariates(model.fun)
  ncon = nconstraints(model)
  nfun = nobs * nvar
  nrow = nfun + ncon
  ncol = nrow

  # index of first constraint
  ind = nfun + 1

  # set drift blocks
  @inbounds for e in 1:nelements(domain)
    p = centroid(domain, e)
    for n in eachindex(drifts)
      f = drifts[n](p)
      for j in (1 + (e - 1) * nvar):(e * nvar), i in (ind + (n - 1) * nvar):(ind + n * nvar - 1)
        LHS[i, j] = ustrip(f) * (i == mod1(j, nvar) + ind + (n - 1) * nvar - 1)
      end
    end
  end
  @inbounds for j in ind:ncol, i in 1:(ind - 1)
    LHS[i, j] = LHS[j, i]
  end

  # set zero block
  @inbounds for j in ind:ncol, i in ind:nrow
    LHS[i, j] = 0
  end

  nothing
end

function rhsconstraints!(fitted::FittedKriging{<:UniversalKriging}, gₒ)
  RHS = fitted.state.RHS
  drifts = fitted.model.drifts
  nvar = nvariates(fitted.model.fun)
  nfun = fitted.state.nfun
  ncol = nvar

  # index of first constraint
  ind = nfun + 1

  # target point
  pₒ = centroid(gₒ)

  # set drift blocks
  @inbounds for n in eachindex(drifts)
    f = drifts[n](pₒ)
    for j in 1:ncol, i in (ind + (n - 1) * ncol):(ind + n * ncol - 1)
      RHS[i, j] = ustrip(f) * (i == j + ind + (n - 1) * ncol - 1)
    end
  end

  nothing
end
