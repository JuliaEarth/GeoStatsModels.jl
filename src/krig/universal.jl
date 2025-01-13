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
  deg::Int
  dim::Int
  expmat::Matrix{Int}

  function UniversalKriging{F}(f, deg, dim) where {F<:GeoStatsFunction}
    @assert deg ≥ 0 "degree must be nonnegative"
    @assert dim > 0 "dimension must be positive"
    expmat = UKexps(deg, dim)
    new(f, deg, dim, expmat)
  end
end

UniversalKriging(f, deg, dim) = UniversalKriging{typeof(f)}(f, deg, dim)

function UKexps(deg::Int, dim::Int)
  # multinomial expansion
  expmats = [hcat(collect(multiexponents(dim, d))...) for d in 0:deg]
  expmat = hcat(expmats...)

  # sort expansion for better conditioned Kriging matrices
  sorted = sortperm(vec(maximum(expmat, dims=1)), rev=true)

  expmat[:, sorted]
end

nconstraints(model::UniversalKriging, nvar::Int) = size(model.expmat, 2)

function set_constraints_lhs!(model::UniversalKriging, LHS::AbstractMatrix, nvar::Int, domain)
  expmat = model.expmat
  nobs = nelements(domain)
  nterms = size(expmat, 2)

  # set polynomial drift blocks
  for i in 1:nobs
    x = CoordRefSystems.raw(coords(centroid(domain, i)))
    for j in 1:nterms
      LHS[nobs + j, i] = prod(x .^ expmat[:, j])
      LHS[i, nobs + j] = LHS[nobs + j, i]
    end
  end

  # set zero block
  LHS[(nobs + 1):end, (nobs + 1):end] .= zero(eltype(LHS))

  nothing
end

function set_constraints_rhs!(fitted::FittedKriging{<:UniversalKriging}, gₒ)
  RHS = fitted.state.RHS
  expmat = fitted.model.expmat
  nobs = nrow(fitted.state.data)
  nterms = size(expmat, 2)

  # set polynomial drift
  xₒ = CoordRefSystems.raw(coords(centroid(gₒ)))
  for j in 1:nterms
    RHS[nobs + j] = prod(xₒ .^ expmat[:, j])
  end

  nothing
end
