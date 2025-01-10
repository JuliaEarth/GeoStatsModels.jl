# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    UniversalKriging(f, degree, dim)

Universal Kriging with geostatistical function `f` and
polynomial of given `degree` on `dim` coordinates.

### Notes

* [`OrdinaryKriging`](@ref) is recovered for 0th degree polynomial
* For non-polynomial mean, see [`ExternalDriftKriging`](@ref)
"""
struct UniversalKriging{F<:GeoStatsFunction} <: KrigingModel
  f::F
  degree::Int
  dim::Int
  exponents::Matrix{Int}

  function UniversalKriging{F}(f, degree, dim) where {F<:GeoStatsFunction}
    @assert degree ≥ 0 "degree must be nonnegative"
    @assert dim > 0 "dimension must be positive"
    exponents = UKexps(degree, dim)
    new(f, degree, dim, exponents)
  end
end

UniversalKriging(f, degree, dim) = UniversalKriging{typeof(f)}(f, degree, dim)

function UKexps(degree::Int, dim::Int)
  # multinomial expansion
  expmats = [hcat(collect(multiexponents(dim, d))...) for d in 0:degree]
  exponents = hcat(expmats...)

  # sort expansion for better conditioned Kriging matrices
  sorted = sortperm(vec(maximum(exponents, dims=1)), rev=true)

  exponents[:, sorted]
end

nconstraints(model::UniversalKriging) = size(model.exponents, 2)

function set_constraints_lhs!(model::UniversalKriging, LHS::AbstractMatrix, domain)
  exponents = model.exponents
  nobs = nelements(domain)
  nterms = size(exponents, 2)

  # set polynomial drift blocks
  for i in 1:nobs
    x = CoordRefSystems.raw(coords(centroid(domain, i)))
    for j in 1:nterms
      LHS[nobs + j, i] = prod(x .^ exponents[:, j])
      LHS[i, nobs + j] = LHS[nobs + j, i]
    end
  end

  # set zero block
  LHS[(nobs + 1):end, (nobs + 1):end] .= zero(eltype(LHS))

  nothing
end

function set_constraints_rhs!(fitted::FittedKriging{<:UniversalKriging}, gₒ)
  exponents = fitted.model.exponents
  RHS = fitted.state.RHS
  nobs = nrow(fitted.state.data)
  nterms = size(exponents, 2)

  # set polynomial drift
  xₒ = CoordRefSystems.raw(coords(centroid(gₒ)))
  for j in 1:nterms
    RHS[nobs + j] = prod(xₒ .^ exponents[:, j])
  end

  nothing
end
