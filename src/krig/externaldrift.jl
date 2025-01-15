# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    ExternalDriftKriging(fun, drifts)

External Drift Kriging with geostatistical function `fun` and
external `drifts` functions. A drift function `p -> v` maps
a point `p` to a value `v`.

### Notes

* External drift functions should be smooth
* Kriging system with external drift is often unstable
* Include a constant drift (e.g. `p -> 1`) for unbiased estimation
* [`OrdinaryKriging`](@ref) is recovered for `drifts = [p -> 1]`
* For polynomial mean, see [`UniversalKriging`](@ref)
"""
struct ExternalDriftKriging{F<:GeoStatsFunction,D} <: KrigingModel
  fun::F
  drifts::Vector{D}
end

nconstraints(model::ExternalDriftKriging, nvar::Int) = nvar * length(model.drifts)

function set_constraints_lhs!(model::ExternalDriftKriging, LHS::AbstractMatrix, nvar::Int, domain)
  # number of constraints
  ncon = nconstraints(model, nvar)

  # index of first constraint
  ind = size(LHS, 1) - ncon + 1

  # auxiliary variables
  drifts = model.drifts
  ONE = I(nvar)

  # set drift blocks
  for j in 1:nelements(domain)
    p = centroid(domain, j)
    for i in eachindex(drifts)
      F = drifts[i](p) * ONE
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

function set_constraints_rhs!(fitted::FittedKriging{<:ExternalDriftKriging}, gₒ)
  RHS = fitted.state.RHS
  nvar = fitted.state.nvar
  model = fitted.model

  # number of constraints
  ncon = nconstraints(model, nvar)

  # index of first constraint
  ind = size(RHS, 1) - ncon + 1

  # auxiliary variables
  drifts = model.drifts
  ONE = I(nvar)

  # target point
  pₒ = centroid(gₒ)

  # set drift blocks
  for i in eachindex(drifts)
    F = drifts[i](pₒ) * ONE
    RHS[(ind + (i - 1) * nvar):(ind + i * nvar - 1), :] .= F
  end

  nothing
end
