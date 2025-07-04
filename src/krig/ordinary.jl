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

scale(model::OrdinaryKriging, α) = OrdinaryKriging(GeoStatsFunctions.scale(model.fun, α))

nconstraints(model::OrdinaryKriging) = nvariates(model.fun)

function lhsconstraints!(model::OrdinaryKriging, LHS::AbstractMatrix, domain)
  nobs = nelements(domain)
  nvar = nvariates(model.fun)
  ncon = nconstraints(model)
  nfun = nobs * nvar
  nrow = nfun + ncon
  ncol = nrow

  # index of first constraint
  ind = nfun + 1

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
  nvar = nvariates(fitted.model.fun)
  ncon = nconstraints(fitted.model)
  nfun = fitted.state.nfun
  nrow = nfun + ncon
  ncol = nvar

  # index of first constraint
  ind = nfun + 1

  # set identity block
  @inbounds for j in 1:ncol, i in ind:nrow
    RHS[i, j] = (i == j + ind - 1)
  end

  nothing
end
