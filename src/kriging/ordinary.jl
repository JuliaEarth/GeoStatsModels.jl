# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    OrdinaryKriging(γ)
    OrdinaryKriging(data, γ)

Ordinary Kriging with variogram model `γ`.

Optionally, pass the geospatial `data` to the [`fit`](@ref) function.
"""
struct OrdinaryKriging{G<:Variogram} <: KrigingModel
  γ::G
end

nconstraints(::OrdinaryKriging) = 1

function set_constraints_lhs!(::OrdinaryKriging, LHS::AbstractMatrix, domain)
  T = eltype(LHS)
  LHS[end, :] .= one(T)
  LHS[:, end] .= one(T)
  LHS[end, end] = zero(T)
  nothing
end

function set_constraints_rhs!(fitted::FittedKriging{<:OrdinaryKriging}, uₒ)
  RHS = fitted.state.RHS
  RHS[end] = one(eltype(RHS))
  nothing
end
