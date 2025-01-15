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
  # initialize Kriging system
  LHS, RHS, STDSQ, nobs, nvar = initkrig(model, domain(data))

  # factorize LHS
  FLHS = factorize(model, LHS)

  # record Kriging state
  state = KrigingState(data, FLHS, RHS, STDSQ, nobs, nvar)

  FittedKriging(model, state)
end

# initialize Kriging system
function initkrig(model::KrigingModel, domain)
  dom = domain
  fun = model.fun

  # retrieve matrix parameters
  STDSQ, (_, nobs, nvar) = GeoStatsFunctions.matrixparams(fun, dom)
  nfun = nobs * nvar
  ncon = nconstraints(model, nvar)
  nrow = nfun + ncon

  # pre-allocate memory for LHS
  F = Matrix{STDSQ}(undef, nrow, nrow)

  # set main block with pairwise evaluation
  GeoStatsFunctions.pairwise!(F, fun, dom)

  # strip units if necessary
  LHS = ustrip.(F)

  # set blocks of constraints
  lhsconstraints!(model, LHS, nvar, dom)

  # pre-allocate memory for RHS
  RHS = Matrix{eltype(LHS)}(undef, nrow, nvar)

  LHS, RHS, STDSQ, nobs, nvar
end

# factorize LHS of Kriging system with appropriate method
factorize(model::KrigingModel, LHS) = factorize(model.fun, LHS)

# enforce Bunch-Kaufman factorization in case of variograms
# as they produce dense symmetric matrices (including constraints)
factorize(::Variogram, LHS) = bunchkaufman(Symmetric(LHS), check=false)

# find appropriate matrix factorization in case of general
# geostatistical functions (e.g. covariances, transiograms)
factorize(::GeoStatsFunction, LHS) = LinearAlgebra.factorize(LHS)

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedKriging, vars, gₒ) = predictmean(fitted, weights(fitted, gₒ), vars)

function predictprob(fitted::FittedKriging, vars, gₒ)
  w = weights(fitted, gₒ)
  μ = predictmean(fitted, w, vars)
  σ² = predictvar(fitted, w)
  Normal(μ, √σ²)
end

predictmean(fitted::FittedKriging, weights::KrigingWeights, vars) = krigmean(fitted, weights, vars)
predictmean(fitted::FittedKriging, weights::KrigingWeights, var::Symbol) =
  first(predictmean(fitted, weights, (var,)))
predictmean(fitted::FittedKriging, weights::KrigingWeights, var::AbstractString) =
  predictmean(fitted, weights, Symbol(var))

function krigmean(fitted::FittedKriging, weights::KrigingWeights, vars)
  d = fitted.state.data
  k = fitted.state.nvar
  λ = weights.λ

  cols = Tables.columns(values(d))
  @inbounds ntuple(k) do j
    sum(1:k) do p
      λₚ = @view λ[p:k:end, j]
      zₚ = Tables.getcolumn(cols, vars[p])
      sum(i -> λₚ[i] * zₚ[i], eachindex(λₚ, zₚ))
    end
  end
end

predictvar(fitted::FittedKriging, weights::KrigingWeights) =
  krigvar(fitted.model.fun, weights, fitted.state)

function krigvar(::Variogram, weights::KrigingWeights, state)
  RHS = state.RHS
  V = state.STDSQ

  # weights and Lagrange multipliers
  λ = weights.λ
  ν = weights.ν

  # compute RHS' * [λ; ν] efficiently
  n = size(λ, 1)
  Σλ = transpose(@view RHS[1:n, :]) * λ
  Σν = transpose(@view RHS[(n + 1):end, :]) * ν
  σ² = tr(Σλ) + tr(Σν)

  # treat numerical issues
  max(zero(V), V(σ²))
end

function weights(fitted::FittedKriging, gₒ)
  dom = domain(fitted.state.data)
  nobs = fitted.state.nobs
  nvar = fitted.state.nvar
  nfun = nobs * nvar

  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(dom))

  rhs!(fitted, gₒ′)

  # solve Kriging system
  s = fitted.state.LHS \ fitted.state.RHS

  λ = @view s[1:nfun, :]
  ν = @view s[(nfun + 1):end, :]

  KrigingWeights(λ, ν)
end

function rhs!(fitted::FittedKriging, gₒ)
  dom = domain(fitted.state.data)
  fun = fitted.model.fun
  RHS = fitted.state.RHS
  nobs = fitted.state.nobs
  nvar = fitted.state.nvar

  # set RHS with function evaluation
  @inbounds for i in 1:nobs
    gᵢ = dom[i]
    RHS[(i-1)*nvar+1:i*nvar,1:nvar] .= ustrip.(fun(gᵢ, gₒ))
  end

  rhsconstraints!(fitted, gₒ)
end

# the following functions are implemented by
# all variants of Kriging (e.g., SimpleKriging)
function nconstraints end
function lhsconstraints! end
function rhsconstraints! end

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
