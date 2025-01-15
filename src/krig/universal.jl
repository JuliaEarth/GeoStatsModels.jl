# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    UniversalKriging(fun, degree, dim)

Universal Kriging with geostatistical function `fun` and
polynomial of given `degree` on `dim` coordinates.

### Notes

* [`OrdinaryKriging`](@ref) is recovered for 0th degree polynomial
* For non-polynomial mean, see [`ExternalDriftKriging`](@ref)
"""
struct UniversalKriging{F<:GeoStatsFunction} <: KrigingModel
  fun::F
  deg::Int
  dim::Int
  pow::Matrix{Int}

  function UniversalKriging{F}(fun, deg, dim) where {F<:GeoStatsFunction}
    @assert deg ≥ 0 "degree must be nonnegative"
    @assert dim > 0 "dimension must be positive"
    pow = powermatrix(deg, dim)
    new(fun, deg, dim, pow)
  end
end

UniversalKriging(fun, deg, dim) = UniversalKriging{typeof(fun)}(fun, deg, dim)

function powermatrix(deg::Int, dim::Int)
  # multinomial expansion up to given degree
  pow = reduce(hcat, stack(multiexponents(dim, d)) for d in 0:deg)

  # sort for better conditioned Kriging matrices
  inds = sortperm(vec(maximum(pow, dims=1)), rev=true)

  pow[:, inds]
end

nconstraints(model::UniversalKriging, nvar::Int) = nvar * size(model.pow, 2)

function lhsconstraints!(model::UniversalKriging, LHS::AbstractMatrix, nvar::Int, domain)
  # number of constraints
  ncon = nconstraints(model, nvar)

  # index of first constraint
  ind = size(LHS, 1) - ncon + 1

  # auxiliary variables
  pow = model.pow
  ONE = I(nvar)

  # set polynomial drift blocks
  for j in 1:nelements(domain)
    p = centroid(domain, j)
    x = CoordRefSystems.raw(coords(p))
    for i in 1:size(pow, 2)
      F = prod(x .^ pow[:, i]) * ONE
      LHS[(ind + (i - 1) * nvar):(ind + i * nvar - 1), ((j - 1) * nvar + 1):(j * nvar)] .= F
    end
  end
  for j in ind:size(LHS, 2)
    for i in 1:(ind - 1)
      LHS[i, j] = LHS[j, i]
    end
  end

  # set zero block
  LHS[ind:end, ind:end] .= zero(eltype(LHS))

  nothing
end

function rhsconstraints!(fitted::FittedKriging{<:UniversalKriging}, gₒ)
  RHS = fitted.state.RHS
  nvar = fitted.state.nvar
  model = fitted.model

  # number of constraints
  ncon = nconstraints(model, nvar)

  # index of first constraint
  ind = size(RHS, 1) - ncon + 1

  # auxiliary variables
  pow = model.pow
  ONE = I(nvar)

  # target coordinates
  pₒ = centroid(gₒ)
  xₒ = CoordRefSystems.raw(coords(pₒ))

  # set polynomial drift blocks
  for i in 1:size(pow, 2)
    F = prod(xₒ .^ pow[:, i]) * ONE
    RHS[(ind + (i - 1) * nvar):(ind + i * nvar - 1), :] .= F
  end

  nothing
end
