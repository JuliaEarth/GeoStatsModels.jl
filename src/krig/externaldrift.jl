# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    ExternalDriftKriging(γ, drifts)

External Drift Kriging with variogram model `γ` and
external `drifts` functions. A drift function `p -> v`
maps a point `p` to a value `v`.

### Notes

* External drift functions should be smooth
* Kriging system with external drift is often unstable
* Include a constant drift (e.g. `p -> 1`) for unbiased estimation
* [`OrdinaryKriging`](@ref) is recovered for `drifts = [p -> 1]`
* For polynomial mean, see [`UniversalKriging`](@ref)
"""
struct ExternalDriftKriging{G<:Variogram,D} <: KrigingModel
  γ::G
  drifts::Vector{D}
end

nconstraints(model::ExternalDriftKriging) = length(model.drifts)

function set_constraints_lhs!(model::ExternalDriftKriging, LHS::AbstractMatrix, domain)
  drifts = model.drifts
  ndrifts = length(drifts)
  nobs = nelements(domain)

  # set external drift blocks
  for i in 1:nobs
    p = centroid(domain, i)
    for j in 1:ndrifts
      LHS[nobs + j, i] = drifts[j](p)
      LHS[i, nobs + j] = LHS[nobs + j, i]
    end
  end

  # set zero block
  LHS[(nobs + 1):end, (nobs + 1):end] .= zero(eltype(LHS))

  nothing
end

function set_constraints_rhs!(fitted::FittedKriging{<:ExternalDriftKriging}, gₒ)
  drifts = fitted.model.drifts
  RHS = fitted.state.RHS
  nobs = nrow(fitted.state.data)

  # set external drift
  pₒ = centroid(gₒ)
  for (j, m) in enumerate(drifts)
    RHS[nobs + j] = m(pₒ)
  end

  nothing
end
