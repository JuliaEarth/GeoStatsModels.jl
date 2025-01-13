# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    KrigingModel

A Kriging model (e.g. Simple Kriging, Ordinary Kriging).
"""
abstract type KrigingModel <: GeoStatsModel end

"""
    KrigingState(data, LHS, RHS, STDSQ)

A Kriging state stores information needed
to perform estimation at any given geometry.
"""
mutable struct KrigingState{D<:AbstractGeoTable,F<:Factorization,A,S}
  data::D
  LHS::F
  RHS::A
  STDSQ::S
  nobs::Int
  nvar::Int
end

"""
    KrigingWeights(λ, ν)

An object storing Kriging weights `λ` and Lagrange multipliers `ν`.
"""
struct KrigingWeights{A}
  λ::A
  ν::A
end

"""
    FittedKriging(model, state)

An object that can be used for making predictions using the
parameters in Kriging `model` and the current Kriging `state`.
"""
struct FittedKriging{M<:KrigingModel,S<:KrigingState}
  model::M
  state::S
end

status(fitted::FittedKriging) = issuccess(fitted.state.LHS)

#--------------
# FITTING STEP
#--------------

function fit(model::KrigingModel, data)
  # build Kriging system
  LHS, RHS, STDSQ, nobs, nvar = initkrig(model, domain(data))

  # factorize LHS
  FLHS = factorize(model, LHS)

  # record Kriging state
  state = KrigingState(data, FLHS, RHS, STDSQ, nobs, nvar)

  FittedKriging(model, state)
end

"""
    initkrig(model, domain)

Initialize the linear system for the Kriging `model` and `domain`.
"""
function initkrig(model::KrigingModel, domain)
  # geostatistical function and domain
  fun = model.f
  dom = domain

  # retrieve matrix parameters
  STDSQ, (_, nobs, nvar) = GeoStatsFunctions.matrixparams(fun, dom)
  nfun = nobs * nvar
  ncon = nconstraints(model)
  nrow = nfun + ncon

  # pre-allocate memory for LHS
  F = Matrix{STDSQ}(undef, nrow, nrow)

  # set main block with pairwise evaluation
  GeoStatsFunctions.pairwise!(F, fun, dom)

  # strip units if necessary
  LHS = ustrip.(F)

  # set blocks of constraints
  set_constraints_lhs!(model, LHS, dom)

  # pre-allocate memory for RHS
  RHS = Matrix{eltype(LHS)}(undef, nrow, nvar)

  LHS, RHS, STDSQ, nobs, nvar
end

"""
    nconstraints(model)

Return number of constraints for Kriging `model`.
"""
function nconstraints end

"""
    set_constraints_lhs!(model, LHS, X)

Set constraints in LHS of Kriging system.
"""
function set_constraints_lhs! end

"""
    factorize(model, LHS)

Factorize LHS of Kriging system with appropriate
factorization method.
"""
factorize(::KrigingModel, LHS) = bunchkaufman(Symmetric(LHS), check=false)

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedKriging, var, gₒ) = predictmean(fitted, weights(fitted, gₒ), var)

function predictprob(fitted::FittedKriging, var, gₒ)
  w = weights(fitted, gₒ)
  μ = predictmean(fitted, w, var)
  σ² = predictvar(fitted, w)
  Normal(μ, √σ²)
end

"""
    predictmean(fitted, weights, vars)

Posterior mean of `fitted` Kriging model with
given `weights` for variables `vars`.
"""
predictmean(fitted::FittedKriging, weights::KrigingWeights, vars) = krigmean(fitted, weights, vars)
predictmean(fitted::FittedKriging, weights::KrigingWeights, var::Symbol) =
  first(predictmean(fitted, weights, (var,)))
predictmean(fitted::FittedKriging, weights::KrigingWeights, var::AbstractString) =
  predictmean(fitted, weights, Symbol(var))

"""
    krigmean(fitted, weights, vars)

Actual implementation of `predictmean` for a
`fitted` Kriging model with given `weights`
for multiple variables `vars`.
"""
function krigmean(fitted::FittedKriging, weights::KrigingWeights, vars)
  d = fitted.state.data
  k = fitted.state.nvar
  λ = weights.λ

  cols = Tables.columns(values(d))
  @inbounds ntuple(k) do j
    λⱼ = @view λ[j:k:end, j]
    zⱼ = Tables.getcolumn(cols, vars[j])
    sum(i -> λⱼ[i] * zⱼ[i], eachindex(λⱼ, zⱼ))
  end
end

"""
    predictvar(fitted, weights)

Posterior variance of `fitted` Kriging model with
given `weights` for variable `var`.
"""
function predictvar(fitted::FittedKriging, weights::KrigingWeights)
  f = fitted.model.f
  b = fitted.state.RHS
  V = fitted.state.STDSQ
  λ = weights.λ
  ν = weights.ν

  # compute b⋅[λ;ν]
  n = length(λ)
  m = length(b)
  c₁ = view(b, 1:n) ⋅ λ
  c₂ = view(b, (n + 1):m) ⋅ ν
  c = c₁ + c₂

  σ² = isstationary(f) ? sill(f) - V(c) : V(c)

  max(zero(V), σ²)
end

"""
    weights(model, gₒ)

Weights λ (and Lagrange multipliers ν) for the
Kriging `model` at geometry `gₒ`.
"""
function weights(fitted::FittedKriging, gₒ)
  dom = domain(fitted.state.data)
  nobs = fitted.state.nobs
  nvar = fitted.state.nvar
  nfun = nobs * nvar

  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(dom))

  set_rhs!(fitted, gₒ′)

  # solve Kriging system
  s = fitted.state.LHS \ fitted.state.RHS

  λ = @view s[1:nfun, :]
  ν = @view s[(nfun + 1):end, :]

  KrigingWeights(λ, ν)
end

"""
    set_rhs!(model, gₒ)

Set RHS of Kriging system at geometry `gₒ`.
"""
function set_rhs!(fitted::FittedKriging, gₒ)
  fun = fitted.model.f
  dom = domain(fitted.state.data)
  RHS = fitted.state.RHS
  nobs = fitted.state.nobs
  nvar = fitted.state.nvar

  # set RHS with function evaluation
  @inbounds for i in 1:nobs
    gᵢ = dom[i]
    RHS[(i-1)*nvar+1:i*nvar,1:nvar] .= ustrip.(fun(gᵢ, gₒ))
  end

  set_constraints_rhs!(fitted, gₒ)
end

"""
    set_constraints_rhs!(model, xₒ)

Set constraints in RHS of Kriging system.
"""
function set_constraints_rhs! end

# ----------------
# IMPLEMENTATIONS
# ----------------

include("krig/simple.jl")
include("krig/ordinary.jl")
include("krig/universal.jl")
include("krig/externaldrift.jl")

"""
    Kriging(f)

Equivalent to [`OrdinaryKriging`](@ref) with
geostatistical function `f`.

    Kriging(f, μ)

Equivalent to [`SimpleKriging`](@ref) with
geostatistical function `f` and constant mean `μ`.

    Kriging(f, deg, dim)

Equivalent to [`UniversalKriging`](@ref) with
geostatistical function `f` and `deg`-order
polynomial in `dim`-dimensinal space.

    Kriging(f, drifts)

Equivalent to [`ExternalDriftKriging`](@ref) with
geostatistical function `f` and `drifts` functions.

Please check the docstring of corresponding models for more details.
"""
Kriging(f::GeoStatsFunction) = OrdinaryKriging(f)
Kriging(f::GeoStatsFunction, μ::Number) = SimpleKriging(f, μ)
Kriging(f::GeoStatsFunction, deg::Int, dim::Int) = UniversalKriging(f, deg, dim)
Kriging(f::GeoStatsFunction, drifts::AbstractVector) = ExternalDriftKriging(f, drifts)
