# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    UniversalKriging(fun, drifts)

Universal Kriging with geostatistical function `fun` and `drifts`.
A drift is a function `p -> v` maps a point `p` to a unitless value `v`.

    UniversalKriging(fun, deg, dim)

Alternatively, construct monomial `drifts` up to given degree `deg`
for `dim` geospatial coordinates.

### Notes

* Drift functions should be smooth for numerical stability
* Include a constant drift (e.g. `p -> 1`) for unbiased estimation
* [`OrdinaryKriging`](@ref) is recovered with `drifts = [p -> 1]`
"""
struct UniversalKriging{F<:GeoStatsFunction,D} <: KrigingModel
  fun::F
  drifts::D
end

UniversalKriging(fun::GeoStatsFunction, deg::Int, dim::Int) =
  UniversalKriging(fun, monomials(deg, dim))

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

nconstraints(model::UniversalKriging, nvar::Int) = nvar * length(model.drifts)

function lhsconstraints!(model::UniversalKriging, LHS::AbstractMatrix, nvar::Int, domain)
  # number of constraints
  ncon = nconstraints(model, nvar)

  # index of first constraint
  ind = size(LHS, 1) - ncon + 1

  # auxiliary variables
  drifts = model.drifts
  Iₖ = I(nvar)

  # set drift blocks
  @inbounds for j in 1:nelements(domain)
    p = centroid(domain, j)
    for i in eachindex(drifts)
      F = drifts[i](p) * Iₖ
      LHS[(ind + (i - 1) * nvar):(ind + i * nvar - 1), ((j - 1) * nvar + 1):(j * nvar)] .= F
    end
  end
  @inbounds for j in ind:size(LHS, 2)
    for i in 1:(ind - 1)
      LHS[i, j] = LHS[j, i]
    end
  end

  # set zero block
  @inbounds LHS[ind:end, ind:end] .= zero(eltype(LHS))

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
  drifts = model.drifts
  Iₖ = I(nvar)

  # target point
  pₒ = centroid(gₒ)

  # set drift blocks
  @inbounds for i in eachindex(drifts)
    F = drifts[i](pₒ) * Iₖ
    RHS[(ind + (i - 1) * nvar):(ind + i * nvar - 1), :] .= F
  end

  nothing
end
