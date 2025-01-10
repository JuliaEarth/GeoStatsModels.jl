# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    KrigingModel

A Kriging model (e.g. Simple Kriging).
"""
abstract type KrigingModel <: GeoStatsModel end

"""
    KrigingState(data, LHS, RHS, STDSQ)

A Kriging state stores information needed
to perform estimation at any given geometry.
"""
mutable struct KrigingState{D<:AbstractGeoTable,F<:Factorization,T,S}
  data::D
  LHS::F
  RHS::Vector{T}
  STDSQ::S
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
  # geostatistical function and domain
  f = model.f
  D = domain(data)

  # build Kriging system
  LHS = lhs(model, D)
  RHS = Vector{eltype(LHS)}(undef, size(LHS, 1))

  # factorize LHS
  FLHS = factorize(model, LHS)

  # variance (σ²) type
  STDSQ, _ = GeoStatsFunctions.matrixparams(f, D)

  # record Kriging state
  state = KrigingState(data, FLHS, RHS, STDSQ)

  # return fitted model
  FittedKriging(model, state)
end

"""
    lhs(model, domain)

Return LHS of Kriging system for the elements in the `domain`.
"""
function lhs(model::KrigingModel, domain)
  f = model.f
  V, (_, nobs, nvars) = GeoStatsFunctions.matrixparams(f, domain)

  # pre-allocate memory for LHS
  nmat = nobs * nvars
  ncon = nconstraints(model)
  G = Matrix{V}(undef, nmat + ncon, nmat + ncon)

  # set main block with pairwise evaluation
  GeoStatsFunctions.pairwise!(G, f, domain)
  if isstationary(f)
    σ² = sill(f)
    for j in 1:nmat, i in 1:nmat
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
    weights(model, uₒ)

Weights λ (and Lagrange multipliers ν) for the
Kriging `model` at geometry `uₒ`.
"""
function weights(fitted::FittedKriging, uₒ)
  dom = domain(fitted.state.data)
  nobs = nrow(fitted.state.data)

  # adjust CRS of uₒ
  uₒ′ = uₒ |> Proj(crs(dom))

  set_rhs!(fitted, uₒ′)

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
  f = fitted.model.f
  dom = domain(fitted.state.data)
  nel = nelements(dom)
  RHS = fitted.state.RHS

  # set RHS with function evaluation
  g = map(u -> f(u, uₒ), dom)
  RHS[1:nel] .= ustrip.(g)
  if isstationary(f)
    σ² = ustrip(sill(f))
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
geostatistical `f` and `drifts` functions.

Please check the docstring of corresponding models for more details.
"""
Kriging(f) = OrdinaryKriging(f)
Kriging(f, μ::Number) = SimpleKriging(f, μ)
Kriging(f, deg::Int, dim::Int) = UniversalKriging(f, deg, dim)
Kriging(f, drifts::AbstractVector) = ExternalDriftKriging(f, drifts)
