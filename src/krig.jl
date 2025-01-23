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
mutable struct KrigingState{D<:AbstractGeoTable,F,A}
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
struct FittedKriging{M<:KrigingModel,S<:KrigingState} <: FittedGeoStatsModel
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
  FLHS = lhsfactorize(model, LHS)

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

  # adjustments for numerical stability
  lhsadjustments!(LHS, fun, dom)

  # set blocks of constraints
  lhsconstraints!(model, LHS, nvar, dom)

  # pre-allocate memory for RHS
  RHS = similar(LHS, nrow, nvar)

  LHS, RHS, ncon
end

# choose appropriate factorization of LHS
lhsfactorize(model::GeoStatsModel, LHS) = lhsfactorize(model.fun, LHS)

# Bunch-Kaufman factorization in case of dense symmetric matrices
lhsfactorize(::Variogram, LHS) = bunchkaufman!(Symmetric(LHS), check=false)

# LU factorization in case of general square matrices
lhsfactorize(::GeoStatsFunction, LHS) = lu!(LHS, check=false)

# convert variograms into covariances for better numerical stability
function lhsadjustments!(LHS, fun::Variogram, dom)
  # adjustments only possible in stationary case
  isstationary(fun) || return nothing

  # retrieve total sill
  S = ustrip.(sill(fun))

  # retrieve matrix paramaters
  nobs = nelements(dom)
  nvar = size(S, 1)
  nfun = nobs * nvar

  @inbounds for j in 1:nfun, i in 1:nfun
    LHS[i, j] = S[mod1(i, nvar), mod1(j, nvar)] - LHS[i, j]
  end

  nothing
end

# no adjustments in case of general geostatistical functions
lhsadjustments!(LHS, fun, dom) = nothing

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedKriging, var::AbstractString, gₒ) = predict(fitted, Symbol(var), gₒ)

predict(fitted::FittedKriging, vars, gₒ) = predictmean(fitted, weights(fitted, gₒ), vars)

predictprob(fitted::FittedKriging, var::AbstractString, gₒ) = predictprob(fitted, Symbol(var), gₒ)

function predictprob(fitted::FittedKriging, vars, gₒ)
  w = weights(fitted, gₒ)
  μ = predictmean(fitted, w, vars)
  σ² = predictvar(fitted, w, gₒ)
  # https://github.com/JuliaStats/Distributions.jl/issues/1413
  @. Normal(ustrip(μ), √σ²)
end

predictmean(fitted::FittedKriging, weights::KrigingWeights, vars) = krigmean(fitted, weights, vars)
predictmean(fitted::FittedKriging, weights::KrigingWeights, var::Symbol) = first(predictmean(fitted, weights, (var,)))
predictmean(fitted::FittedKriging, weights::KrigingWeights, var::AbstractString) =
  predictmean(fitted, weights, Symbol(var))

function krigmean(fitted::FittedKriging, weights::KrigingWeights, vars)
  d = fitted.state.data
  λ = weights.λ
  k = length(vars)

  @assert size(λ, 2) == k "invalid number of variables for Kriging model"

  cols = Tables.columns(values(d))
  @inbounds map(1:k) do j
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
  σ²₊ = max.(zero(σ²), σ²)

  # treat scalar case
  length(σ²₊) == 1 ? first(σ²₊) : σ²₊
end

krigvar(fun::Variogram, weights::KrigingWeights, RHS, gₒ) = covvar(fun, weights, RHS, gₒ)

krigvar(fun::Covariance, weights::KrigingWeights, RHS, gₒ) = covvar(fun, weights, RHS, gₒ)

function krigvar(t::Transiogram, weights::KrigingWeights, RHS, gₒ)
  # auxiliary variables
  n, k = size(weights.λ)
  p = proportions(t)

  # convert transiograms to covariances
  COV = deepcopy(RHS)
  @inbounds for j in 1:k, i in 1:n
    # Eq. 12 of Carle & Fogg 1996
    COV[i, j] = p[mod1(i, k)] * (COV[i, j] - p[j])
  end

  # compute variance contributions
  Cλ, Cν = wmul(weights, COV)

  # compute cov(0) considering change of support
  Tₒ = t(gₒ, gₒ)
  Cₒ = @inbounds [p[i] * (Tₒ[i, j] - p[j]) for i in 1:k, j in 1:k]

  diag(Cₒ) - diag(Cλ) - diag(Cν)
end

function covvar(fun::GeoStatsFunction, weights::KrigingWeights, RHS, gₒ)
  # auxiliary variables
  k = size(weights.λ, 2)

  # compute variance contributions
  Cλ, Cν = wmul(weights, RHS)

  # compute cov(0) considering change of support
  Cₒ = ustrip.(covzero(fun, gₒ)) * I(k)

  diag(Cₒ) - diag(Cλ) - diag(Cν)
end

covzero(γ::Variogram, gₒ) = isstationary(γ) ? sill(γ) - γ(gₒ, gₒ) : γ(gₒ, gₒ)

covzero(cov::Covariance, gₒ) = cov(gₒ, gₒ)

# compute RHS' * [λ; ν] efficiently
function wmul(weights::KrigingWeights, RHS)
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

  # adjustments for numerical stability
  rhsadjustments!(RHS, fun, dom)

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

# convert variograms into covariances for better numerical stability
function rhsadjustments!(RHS, fun::Variogram, dom)
  # adjustments only possible in stationary case
  isstationary(fun) || return nothing

  # retrieve total sill
  S = ustrip.(sill(fun))

  # retrieve matrix paramaters
  nobs = nelements(dom)
  nvar = size(S, 1)
  nfun = nobs * nvar

  @inbounds for j in 1:nvar, i in 1:nfun
    RHS[i, j] = S[mod1(i, nvar), mod1(j, nvar)] - RHS[i, j]
  end

  nothing
end

# no adjustments in case of general geostatistical functions
rhsadjustments!(RHS, fun, dom) = nothing

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
