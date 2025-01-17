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

function lhsconstraints!(model::OrdinaryKriging, LHS::AbstractMatrix, nvar::Int, domain)
  # number of constraints
  ncon = nconstraints(model, nvar)

  # retrieve size of LHS
  nrow, ncol = size(LHS)

  # index of first constraint
  ind = nrow - ncon + 1

  # set identity blocks
  @inbounds for j in 1:(ind - 1), i in ind:nrow
    LHS[i, j] = (i == mod1(j, nvar) + ind - 1)
  end
  @inbounds for j in ind:ncol, i in 1:(ind - 1)
    LHS[i, j] = LHS[j, i]
  end

  # set zero block
  @inbounds for j in ind:ncol, i in ind:nrow
    LHS[i, j] = 0
  end

  nothing
end

function rhsconstraints!(fitted::FittedKriging{<:OrdinaryKriging}, gₒ)
  RHS = fitted.state.RHS
  ncon = fitted.state.ncon

  # retrieve size of RHS
  nrow, ncol = size(RHS)

  # index of first constraint
  ind = nrow - ncon + 1

  # set identity block
  @inbounds for j in 1:ncol, i in ind:nrow
    RHS[i, j] = (i == j + ind - 1)
  end

  nothing
end
