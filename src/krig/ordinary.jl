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

  # index of first constraint
  ind = size(LHS, 1) - ncon + 1

  # auxiliary variables
  Iₖ = I(nvar)

  # set identity blocks
  @inbounds for j in 1:nvar:(ind - nvar)
    LHS[ind:end, j:(j + nvar - 1)] .= Iₖ
  end
  @inbounds for i in 1:nvar:(ind - nvar)
    LHS[i:(i + nvar - 1), ind:end] .= Iₖ
  end

  # set zero block
  @inbounds LHS[ind:end, ind:end] .= zero(eltype(LHS))

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
