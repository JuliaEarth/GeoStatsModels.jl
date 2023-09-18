# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    KrigingModel

A Kriging model (e.g. Simple Kriging).
"""
abstract type KrigingModel <: GeoStatsModel end

"""
    KrigingState(data, LHS, RHS, VARTYPE)

A Kriging state stores information needed
to perform estimation at any given geometry.
"""
mutable struct KrigingState{D<:AbstractGeoTable,F<:Factorization,T,V}
  data::D
  LHS::F
  RHS::Vector{T}
  VARTYPE::V
end

"""
    KrigingWeights(λ, ν)

An object storing Kriging weights `λ` and Lagrange multipliers `ν`.
"""
struct KrigingWeights{T<:Real,A<:AbstractVector{T}}
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
  # variogram and domain
  γ = model.γ
  D = domain(data)

  # build Kriging system
  LHS = lhs(model, D)
  RHS = Vector{eltype(LHS)}(undef, size(LHS, 1))

  # factorize LHS
  FLHS = factorize(model, LHS)

  # variance type
  VARTYPE = Variography.result_type(γ, first(D), first(D))

  # record Kriging state
  state = KrigingState(data, FLHS, RHS, VARTYPE)

  # return fitted model
  FittedKriging(model, state)
end

"""
    lhs(model, domain)

Return LHS of Kriging system for the elements in the `domain`.
"""
function lhs(model::KrigingModel, domain)
  γ = model.γ
  nobs = nelements(domain)
  ncon = nconstraints(model)

  # pre-allocate memory for LHS
  u = first(domain)
  V² = Variography.result_type(γ, u, u)
  m = nobs + ncon
  G = Matrix{V²}(undef, m, m)

  # set variogram/covariance block
  Variography.pairwise!(G, γ, domain)
  if isstationary(γ)
    σ² = sill(γ)
    for j in 1:nobs, i in 1:nobs
      @inbounds G[i, j] = σ² - G[i, j]
    end
  end

  # strip units if necessary
  LHS = ustrip.(G)

  # set blocks of constraints
  set_constraints_lhs!(model, LHS, domain)

  LHS
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

predict(fitted::FittedKriging, var, uₒ) = predictmean(fitted, weights(fitted, uₒ), var)

function predictprob(fitted::FittedKriging, var, uₒ)
  w = weights(fitted, uₒ)
  μ = predictmean(fitted, w, var)
  σ² = predictvar(fitted, w)
  Normal(μ, √σ²)
end

"""
    predictmean(fitted, var, weights)

Posterior mean of `fitted` Kriging model for variable `var`
with Kriging `weights`.
"""
function predictmean(fitted::FittedKriging, weights::KrigingWeights, var)
  d = fitted.state.data
  c = Tables.columns(values(d))
  z = Tables.getcolumn(c, var)
  λ = weights.λ
  sum(i -> λ[i] * z[i], eachindex(λ, z))
end

"""
    predictvar(fitted, var, weights)

Posterior variance of `fitted` Kriging model for variable `var`
with Kriging `weights`.
"""
function predictvar(fitted::FittedKriging, weights::KrigingWeights)
  γ = fitted.model.γ
  b = fitted.state.RHS
  V² = fitted.state.VARTYPE
  λ = weights.λ
  ν = weights.ν

  # compute b⋅[λ;ν]
  n = length(λ)
  m = length(b)
  c₁ = view(b, 1:n) ⋅ λ
  c₂ = view(b, (n + 1):m) ⋅ ν
  c = c₁ + c₂

  σ² = isstationary(γ) ? sill(γ) - V²(c) : V²(c)

  max(zero(V²), σ²)
end

"""
    weights(model, uₒ)

Weights λ (and Lagrange multipliers ν) for the
Kriging `model` at geometry `uₒ`.
"""
function weights(fitted::FittedKriging, uₒ)
  nobs = nrow(fitted.state.data)

  set_rhs!(fitted, uₒ)

  # solve Kriging system
  s = fitted.state.LHS \ fitted.state.RHS

  λ = view(s, 1:nobs)
  ν = view(s, (nobs + 1):length(s))

  KrigingWeights(λ, ν)
end

"""
    set_rhs!(model, uₒ)

Set RHS of Kriging system at geometry `uₒ`.
"""
function set_rhs!(fitted::FittedKriging, uₒ)
  γ = fitted.model.γ
  dom = domain(fitted.state.data)
  nel = nelements(dom)
  RHS = fitted.state.RHS

  # RHS variogram/covariance
  g = map(u -> γ(u, uₒ), dom)
  RHS[1:nel] .= ustrip.(g)
  if isstationary(γ)
    σ² = ustrip(sill(γ))
    RHS[1:nel] .= σ² .- RHS[1:nel]
  end

  set_constraints_rhs!(fitted, uₒ)
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
    Kriging(γ)

Equivalent to [`OrdinaryKriging`](@ref) with variogram `γ`.

    Kriging(γ, μ)

Equivalent to [`SimpleKriging`](@ref) with variogram `γ` and
constant mean `μ`.

    Kriging(γ, deg, dim)

Equivalent to [`UniversalKriging`](@ref) with variogram `γ` and
`deg`-order polynomial in `dim`-dimensinal space.

    Kriging(γ, drifts)

Equivalent to [`ExternalDriftKriging`](@ref) with variogram `γ` and
`drifts` functions.

Please check the docstring of corresponding models for more details.
"""
Kriging(γ) = OrdinaryKriging(γ)
Kriging(γ, μ::Number) = SimpleKriging(γ, μ)
Kriging(γ, deg::Int, dim::Int) = UniversalKriging(γ, deg, dim)
Kriging(γ, drifts::AbstractVector) = ExternalDriftKriging(γ, drifts)
