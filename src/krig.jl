# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    KrigingModel

A Kriging model (e.g. Simple Kriging, Ordinary Kriging).
"""
abstract type KrigingModel <: GeoStatsModel end

"""
    KrigingState(data, LHS, RHS, ncon)

A Kriging state stores information needed
to perform estimation at any given geometry.
"""
mutable struct KrigingState{D<:AbstractGeoTable,F<:Factorization,A}
  data::D
  LHS::F
  RHS::A
  ncon::Int
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
  LHS, RHS, ncon, nvar = initkrig(model, domain(data))

  # factorize LHS
  FLHS = factorize(model, LHS)

  # record Kriging state
  state = KrigingState(data, FLHS, RHS, ncon, nvar)

  FittedKriging(model, state)
end

# initialize Kriging system
function initkrig(model::KrigingModel, domain)
  fun = model.fun
  dom = domain

  # retrieve matrix parameters
  V, (_, nobs, nvar) = GeoStatsFunctions.matrixparams(fun, dom)
  nfun = nobs * nvar
  ncon = nconstraints(model, nvar)
  nrow = nfun + ncon

  # pre-allocate memory for LHS
  F = Matrix{V}(undef, nrow, nrow)

  # set main block with pairwise evaluation
  GeoStatsFunctions.pairwise!(F, fun, dom)

  # strip units if necessary
  LHS = ustrip.(F)

  # set blocks of constraints
  lhsconstraints!(model, LHS, nvar, dom)

  # pre-allocate memory for RHS
  RHS = Matrix{eltype(LHS)}(undef, nrow, nvar)

  LHS, RHS, ncon, nvar
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
  σ² = predictvar(fitted, w, gₒ)
  Normal(μ, √σ² * unit(μ))
end

predictmean(fitted::FittedKriging, weights::KrigingWeights, vars) = krigmean(fitted, weights, vars)
predictmean(fitted::FittedKriging, weights::KrigingWeights, var::Symbol) =
  first(predictmean(fitted, weights, (var,)))
predictmean(fitted::FittedKriging, weights::KrigingWeights, var::AbstractString) =
  predictmean(fitted, weights, Symbol(var))

function krigmean(fitted::FittedKriging, weights::KrigingWeights, vars)
  d = fitted.state.data
  λ = weights.λ
  k = length(vars)

  @assert size(λ, 2) == k "invalid number of variables for Kriging model"

  cols = Tables.columns(values(d))
  @inbounds ntuple(k) do j
    sum(1:k) do p
      λₚ = @view λ[p:k:end, j]
      zₚ = Tables.getcolumn(cols, vars[p])
      sum(i -> λₚ[i] * zₚ[i], eachindex(λₚ, zₚ))
    end
  end
end

function predictvar(fitted::FittedKriging, weights::KrigingWeights, gₒ)
  RHS = fitted.state.RHS
  fun = fitted.model.fun

  # variance formula for given function
  σ² = krigvar(fun, weights, RHS, gₒ)

  # treat numerical issues
  max(zero(σ²), σ²)
end

function krigvar(::Variogram, weights::KrigingWeights, RHS, gₒ)
  # weights and Lagrange multipliers
  λ = weights.λ
  ν = weights.ν
  n = size(λ, 1)

  # compute RHS' * [λ; ν] efficiently
  Σλ = transpose(@view RHS[1:n, :]) * λ
  Σν = transpose(@view RHS[(n + 1):end, :]) * ν

  tr(Σλ) + tr(Σν)
end

function krigvar(cov::Covariance, weights::KrigingWeights, RHS, gₒ)
  # weights and Lagrange multipliers
  λ = weights.λ
  ν = weights.ν
  n = size(λ, 1)

  # compute RHS' * [λ; ν] efficiently
  Cλ = transpose(@view RHS[1:n, :]) * λ
  Cν = transpose(@view RHS[(n + 1):end, :]) * ν

  # compute cov(0) considering change of support
  Cₒ = cov(gₒ, gₒ)

  tr(Cₒ) - tr(Cλ) - tr(Cν)
end

function weights(fitted::FittedKriging, gₒ)
  LHS = fitted.state.LHS
  RHS = fitted.state.RHS
  ncon = fitted.state.ncon
  dom = domain(fitted.state.data)

  # index of first constraint
  ind = size(LHS, 1) - ncon + 1

  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(dom))

  # set RHS of Kriging system
  rhs!(fitted, gₒ′)

  # solve Kriging system
  w = LHS \ RHS

  # split weights and Lagrange multipliers
  λ = @view w[begin:(ind - 1), :]
  ν = @view w[ind:end, :]

  KrigingWeights(λ, ν)
end

function rhs!(fitted::FittedKriging, gₒ)
  RHS = fitted.state.RHS
  fun = fitted.model.fun
  dom = domain(fitted.state.data)
  nobs = nelements(dom)
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

"""
    Kriging(f)

Equivalent to [`OrdinaryKriging`](@ref).

    Kriging(f, μ)

Equivalent to [`SimpleKriging`](@ref).

    Kriging(f, deg, dim)

Equivalent to [`UniversalKriging`](@ref).

    Kriging(f, drifts)

Equivalent to [`UniversalKriging`](@ref).

Please check the docstring of corresponding models for more details.
"""
Kriging(f::GeoStatsFunction) = OrdinaryKriging(f)
Kriging(f::GeoStatsFunction, μ::Number) = SimpleKriging(f, μ)
Kriging(f::GeoStatsFunction, deg::Int, dim::Int) = UniversalKriging(f, deg, dim)
Kriging(f::GeoStatsFunction, drifts::AbstractVector) = UniversalKriging(f, drifts)
