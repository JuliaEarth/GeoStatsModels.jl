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
  LHS, RHS, ncon = initkrig(model, domain(data))

  # factorize LHS
  FLHS = factorize(model, LHS)

  # record Kriging state
  state = KrigingState(data, FLHS, RHS, ncon)

  FittedKriging(model, state)
end

# initialize Kriging system
function initkrig(model::KrigingModel, domain)
  fun = model.fun
  dom = domain

  # retrieve matrix parameters
  V, (_, nobs, nvar) = GeoStatsFunctions.matrixparams(fun, dom)
  ncon = nconstraints(model, nvar)
  nrow = nobs * nvar + ncon

  # pre-allocate memory for LHS
  LHS = Matrix{V}(undef, nrow, nrow)

  # set main block with pairwise evaluation
  GeoStatsFunctions.pairwise!(LHS, fun, dom)

  # set blocks of constraints
  lhsconstraints!(model, LHS, nvar, dom)

  # pre-allocate memory for RHS
  RHS = similar(LHS, nrow, nvar)

  LHS, RHS, ncon
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
  @. Normal(μ, √σ² * unit(μ))
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
  # compute variance contributions
  Γλ, Γν = weightedrhs(weights, RHS)

  diag(Γλ) + diag(Γν)
end

function krigvar(cov::Covariance, weights::KrigingWeights, RHS, gₒ)
  # compute variance contributions
  Cλ, Cν = weightedrhs(weights, RHS)

  # compute cov(0) considering change of support
  Cₒ = cov(gₒ, gₒ)

  diag(Cₒ) - diag(Cλ) - diag(Cν)
end

# compute RHS' * [λ; ν] efficiently
function weightedrhs(weights::KrigingWeights, RHS)
  λ = weights.λ
  ν = weights.ν
  n = size(λ, 1)
  b = transpose(RHS)
  Kλ = (@view b[:, 1:n]) * λ
  Kν = (@view b[:, (n + 1):end]) * ν
  
  Kλ, Kν
end

function weights(fitted::FittedKriging, gₒ)
  LHS = fitted.state.LHS
  RHS = fitted.state.RHS
  ncon = fitted.state.ncon
  dom = domain(fitted.state.data)
  fun = fitted.model.fun

  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(dom))

  # set main blocks with pairwise evaluation
  GeoStatsFunctions.pairwise!(RHS, fun, dom, [gₒ′])

  # set blocks of constraints
  rhsconstraints!(fitted, gₒ′)

  # solve Kriging system
  W = LHS \ RHS

  # index of first constraint
  ind = size(LHS, 1) - ncon + 1

  # split weights and Lagrange multipliers
  λ = @view W[begin:(ind - 1), :]
  ν = @view W[ind:end, :]

  KrigingWeights(λ, ν)
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
Kriging(f::GeoStatsFunction, μ::AbstractVector{<:Number}) = SimpleKriging(f, μ)
Kriging(f::GeoStatsFunction, deg::Int, dim::Int) = UniversalKriging(f, deg, dim)
Kriging(f::GeoStatsFunction, drifts::AbstractVector) = UniversalKriging(f, drifts)
