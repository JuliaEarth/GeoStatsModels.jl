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
  pow::Matrix{Int}

  function UniversalKriging{F}(f, deg, dim) where {F<:GeoStatsFunction}
    @assert deg ≥ 0 "degree must be nonnegative"
    @assert dim > 0 "dimension must be positive"
    pow = powermatrix(deg, dim)
    new(f, deg, dim, pow)
  end
end

UniversalKriging(f, deg, dim) = UniversalKriging{typeof(f)}(f, deg, dim)

function powermatrix(deg::Int, dim::Int)
  # multinomial expansion up to given degree
  pow = reduce(hcat, stack(multiexponents(dim, d)) for d in 0:deg)

  # sort for better conditioned Kriging matrices
  inds = sortperm(vec(maximum(pow, dims=1)), rev=true)

  pow[:, inds]
end

nconstraints(model::UniversalKriging, nvar::Int) = size(model.pow, 2)

function set_constraints_lhs!(model::UniversalKriging, LHS::AbstractMatrix, nvar::Int, domain)
  pow = model.pow
  nobs = nelements(domain)
  nterms = size(pow, 2)

  # set polynomial drift blocks
  for i in 1:nobs
    pᵢ = centroid(domain, i)
    xᵢ = CoordRefSystems.raw(coords(pᵢ))
    for j in 1:nterms
      LHS[nobs + j, i] = prod(xᵢ .^ pow[:, j])
      LHS[i, nobs + j] = LHS[nobs + j, i]
    end
  end

  # set zero block
  LHS[(nobs + 1):end, (nobs + 1):end] .= zero(eltype(LHS))

  nothing
end

function set_constraints_rhs!(fitted::FittedKriging{<:UniversalKriging}, gₒ)
  RHS = fitted.state.RHS
  pow = fitted.model.pow
  nobs = nrow(fitted.state.data)
  nterms = size(pow, 2)

  # set polynomial drift
  pₒ = centroid(gₒ)
  xₒ = CoordRefSystems.raw(coords(pₒ))
  for j in 1:nterms
    RHS[nobs + j] = prod(xₒ .^ pow[:, j])
  end

  nothing
end
