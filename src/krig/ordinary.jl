# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    OrdinaryKriging(fun)

Ordinary Kriging with geostatistical function `fun`.
"""
struct OrdinaryKriging{F<:GeoStatsFunction} <: KrigingModel
  fun::F
end

nconstraints(::OrdinaryKriging, nvar::Int) = nvar

function set_constraints_lhs!(model::OrdinaryKriging, LHS::AbstractMatrix, nvar::Int, domain)
  # number of constraints
  ncon = nconstraints(model, nvar)

  # index of first constraint
  ind = size(LHS, 1) - ncon + 1

  # auxiliary variables
  ONE = I(nvar)

  # set identity blocks
  for j in 1:nvar:(ind - nvar)
    LHS[ind:end, j:(j + nvar - 1)] .= ONE
  end
  for i in 1:nvar:(ind - nvar)
    LHS[i:(i + nvar - 1), ind:end] .= ONE
  end

  # set zero block
  LHS[ind:end, ind:end] .= zero(eltype(LHS))

  nothing
end

function set_constraints_rhs!(fitted::FittedKriging{<:OrdinaryKriging}, gₒ)
  RHS = fitted.state.RHS
  nvar = fitted.state.nvar
  model = fitted.model

  # number of constraints
  ncon = nconstraints(model, nvar)

  # index of first constraint
  ind = size(RHS, 1) - ncon + 1

  # auxiliary variables
  ONE = I(nvar)

  # set identity block
  RHS[ind:end, :] .= ONE

  nothing
end
