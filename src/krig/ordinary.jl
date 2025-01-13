# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    OrdinaryKriging(f)

Ordinary Kriging with geostatistical function `f`.
"""
struct OrdinaryKriging{F<:GeoStatsFunction} <: KrigingModel
  f::F
end

nconstraints(::OrdinaryKriging, nvar::Int) = nvar

function set_constraints_lhs!(::OrdinaryKriging, LHS::AbstractMatrix, nvar::Int, domain)
  # index of first constraint
  ind = size(LHS, 1) - nvar + 1

  # set bottom block
  for j in 1:nvar:(ind - nvar)
    LHS[ind:end, j:(j + nvar - 1)] .= I(nvar)
  end

  # set right block
  for i in 1:nvar:(ind - nvar)
    LHS[i:(i + nvar - 1), ind:end] .= I(nvar)
  end

  # set bottom-right block
  LHS[ind:end, ind:end] .= zero(eltype(LHS))

  nothing
end

function set_constraints_rhs!(fitted::FittedKriging{<:OrdinaryKriging}, gₒ)
  RHS = fitted.state.RHS
  nvar = fitted.state.nvar

  # index of first constraint
  ind = size(RHS, 1) - nvar + 1

  # set bottom block
  RHS[ind:end, :] .= I(nvar)

  nothing
end
