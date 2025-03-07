# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    KrigingModel

A Kriging model (e.g. Simple Kriging, Ordinary Kriging).
"""
abstract type KrigingModel <: GeoStatsModel end

"""
    KrigingState(data, LHS, RHS, FHS, nfun, miss)

A Kriging state stores information needed
to perform estimation at any given geometry.
"""
mutable struct KrigingState{L,R}
  data::AbstractGeoTable
  LHS::L
  RHS::R
  FHS::Factorization
  nfun::Int
  miss::Vector{Int}
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

status(fitted::FittedKriging) = _status(fitted.state.FHS)

_status(FHS) = issuccess(FHS)
_status(FHS::SVD) = true

#--------------
# FITTING STEP
#--------------

function fit(model::KrigingModel, data)
  # check compatibility of model and data
  checkcompat(model, data)

  # pre-allocate memory for LHS and RHS
  LHS, RHS = prealloc(model, data)

  # set LHS of Kriging system
  nfun, miss = setlhs!(model, LHS, data)

  # factorize LHS
  FHS = lhsfactorize(model, LHS)

  # record Kriging state
  state = KrigingState(data, LHS, RHS, FHS, nfun, miss)

  FittedKriging(model, state)
end

function fit!(fitted::FittedKriging, newdata)
  model = fitted.model
  state = fitted.state

  # check compatibility of data size
  checkdatasize(fitted, newdata)

  # check compatibility of model and data
  checkcompat(model, newdata)

  # update state data
  state.data = newdata

  # set LHS of Kriging system
  state.nfun, state.miss = setlhs!(model, state.LHS, newdata)

  # number of modified rows
  ncon = nconstraints(model)
  nrow = state.nfun + ncon

  # factorize LHS
  VHS = @view state.LHS[1:nrow, 1:nrow]
  state.FHS = lhsfactorize(model, VHS)

  nothing
end

# make sure data is compatible with model
function checkcompat(model::KrigingModel, data)
  fun = model.fun
  nvar = nvariates(fun)
  nfeat = ncol(data) - 1
  if nfeat != nvar
    throw(ArgumentError("$nfeat data column(s) provided to $nvar-variate Kriging model"))
  end
end

# make sure data size is compatible
function checkdatasize(fitted::FittedKriging, data)
  LHS = fitted.state.LHS
  nlhs = size(LHS, 1)
  nobs = nrow(data)
  if nobs > nlhs
    throw(ArgumentError("in-place fit called with $nobs data row(s) and $nlhs maximum size"))
  end
end

# pre-allocate memory for LHS and RHS
function prealloc(model::KrigingModel, data)
  fun = model.fun
  dom = domain(data)

  # retrieve matrix parameters
  nobs = nelements(dom)
  nvar = nvariates(fun)
  ncon = nconstraints(model)
  nfun = nobs * nvar
  nrow = nfun + ncon

  # pre-allocate memory for LHS
  F = fun(dom[1], dom[1])
  V = eltype(ustrip.(F))
  LHS = Matrix{V}(undef, nrow, nrow)

  # pre-allocate memory for RHS
  RHS = similar(LHS, nrow, nvar)

  LHS, RHS
end

# set LHS of Kriging system
function setlhs!(model::KrigingModel, LHS, data)
  fun = model.fun
  dom = domain(data)
  tab = values(data)

  # number of function evaluations
  nobs = nelements(dom)
  nvar = nvariates(fun)
  nfun = nobs * nvar

  # find locations with missing values
  miss = missingindices(tab)

  # set main block with pairwise evaluation
  GeoStatsFunctions.pairwise!(LHS, fun, dom)

  # adjustments for numerical stability
  if isstationary(fun) && !isbanded(fun)
    lhsbanded!(LHS, fun, dom)
  end

  # set blocks of constraints
  lhsconstraints!(model, LHS, dom)

  # knock out entries with missing values
  lhsmissings!(LHS, nfun, miss)

  nfun, miss
end

# choose appropriate factorization of LHS
lhsfactorize(model::GeoStatsModel, LHS) = _lhsfactorize(model.fun, LHS)

# enforce Bunch-Kaufman factorization for symmetric functions
_lhsfactorize(::Variogram, LHS) = bunchkaufman!(Symmetric(LHS), check=false)
_lhsfactorize(::Covariance, LHS) = bunchkaufman!(Symmetric(LHS), check=false)

# enforce SVD factorization for rank-deficient matrices
_lhsfactorize(::Transiogram, LHS) = svd!(LHS)

# choose appropriate factorization for other functions
function _lhsfactorize(fun::GeoStatsFunction, LHS)
  if issymmetric(fun)
    # enforce Bunch-Kaufman factorization
    bunchkaufman!(Symmetric(LHS), check=false)
  else
    # fallback to LU factorization
    lu!(LHS, check=false)
  end
end

# convert LHS into banded matrix
function lhsbanded!(LHS, fun, dom)
  # retrieve total sill
  S = ustrip.(sill(fun))

  # retrieve matrix paramaters
  nobs = nelements(dom)
  nvar = nvariates(fun)
  nfun = nobs * nvar

  @inbounds for j in 1:nfun, i in 1:nfun
    LHS[i, j] = S[mod1(i, nvar), mod1(j, nvar)] - LHS[i, j]
  end

  nothing
end

# find locations with missing values
function missingindices(tab)
  cols = Tables.columns(tab)
  vars = Tables.columnnames(cols)
  nvar = length(vars)

  # find locations with missing values and
  # map to entries of blocks in final matrix
  entries = map(1:nvar) do j
    vals = Tables.getcolumn(cols, vars[j])
    inds = findall(ismissing, vals)
    [nvar * (i - 1) + j for i in inds]
  end

  # sort indices to improve locality
  sort(reduce(vcat, entries))
end

# knock out entries with missing values
function lhsmissings!(LHS, nfun, miss)
  @inbounds for j in miss, i in 1:nfun
    LHS[i, j] = 0
  end
  @inbounds for j in 1:nfun, i in miss
    LHS[i, j] = 0
  end

  nothing
end

#-----------------
# PREDICTION STEP
#-----------------

predict(fitted::FittedKriging, var::AbstractString, gₒ) = predict(fitted, Symbol(var), gₒ)

predict(fitted::FittedKriging, var::Symbol, gₒ) = predictmean(fitted, weights(fitted, gₒ), (var,)) |> first

predict(fitted::FittedKriging, vars, gₒ) = predictmean(fitted, weights(fitted, gₒ), vars)

predictprob(fitted::FittedKriging, var::AbstractString, gₒ) = predictprob(fitted, Symbol(var), gₒ)

function predictprob(fitted::FittedKriging, var::Symbol, gₒ)
  w = weights(fitted, gₒ)
  μ = predictmean(fitted, w, (var,)) |> first
  σ² = predictvar(fitted, w, gₒ) |> first
  # https://github.com/JuliaStats/Distributions.jl/issues/1413
  Normal(ustrip.(μ), √σ²)
end

function predictprob(fitted::FittedKriging, vars, gₒ)
  w = weights(fitted, gₒ)
  μ = predictmean(fitted, w, vars)
  Σ = predictvar(fitted, w, gₒ)
  # https://github.com/JuliaStats/Distributions.jl/issues/1413
  MvNormal(ustrip.(μ), Σ)
end

# ----------
# INTERNALS
# ----------

predictmean(fitted::FittedKriging, weights::KrigingWeights, vars) = krigmean(fitted, weights, vars)

function krigmean(fitted::FittedKriging, weights::KrigingWeights, vars)
  d = fitted.state.data
  λ = weights.λ
  k = size(λ, 2)

  cols = Tables.columns(values(d))
  @inbounds map(1:k) do j
    sum(1:k) do p
      λₚ = @view λ[p:k:end, j]
      zₚ = Tables.getcolumn(cols, vars[p])
      sum(i -> λₚ[i] ⦿ zₚ[i], eachindex(λₚ, zₚ))
    end
  end
end

# handle missing values in linear combination
⦿(λ, z) = λ * z
⦿(λ, z::Missing) = 0

function predictvar(fitted::FittedKriging, weights::KrigingWeights, gₒ)
  model = fitted.model
  RHS = fitted.state.RHS
  nfun = fitted.state.nfun
  ncon = nconstraints(model)
  nrow = nfun + ncon

  # geostatistical function
  fun = model.fun

  # view valid rows of RHS
  VHS = @view RHS[1:nrow, :]

  # covariance formula for given function
  Σ = krigvar(fun, weights, VHS, gₒ)

  # treat numerical issues
  ϵ = eltype(Σ)(1e-10)
  Symmetric(Σ + ϵ * I)
end

function krigvar(fun::GeoStatsFunction, weights::KrigingWeights, RHS, gₒ)
  # auxiliary variables
  k = size(weights.λ, 2)

  # compute variance contributions
  Cλ, Cν = wmul(weights, RHS)

  # compute cov(0) considering change of support
  Cₒ = ustrip.(covzero(fun, gₒ)) * I(k)

  Cₒ - Cλ - Cν
end

function covzero(fun::GeoStatsFunction, gₒ)
  if isstationary(fun) && !isbanded(fun)
    sill(fun) - fun(gₒ, gₒ)
  else
    fun(gₒ, gₒ)
  end
end

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

# solve Kriging system at target geometry
function weights(fitted::FittedKriging, gₒ)
  data = fitted.state.data
  FHS = fitted.state.FHS
  RHS = fitted.state.RHS
  nfun = fitted.state.nfun
  nrow = size(FHS, 1)

  # retrieve domain of data
  dom = domain(data)

  # adjust CRS of gₒ
  gₒ′ = gₒ |> Proj(crs(dom))

  # set RHS of Kriging system
  setrhs!(fitted, RHS, dom, gₒ′)

  # solve Kriging system
  W = FHS \ @view RHS[1:nrow, :]

  # split weights and Lagrange multipliers
  λ = @view W[begin:nfun, :]
  ν = @view W[(nfun+1):end, :]

  KrigingWeights(λ, ν)
end

# set RHS of Kriging system
function setrhs!(fitted::FittedKriging, RHS, dom, gₒ)
  fun = fitted.model.fun
  miss = fitted.state.miss

  # set main blocks with pairwise evaluation
  GeoStatsFunctions.pairwise!(RHS, fun, dom, [gₒ])

  # adjustments for numerical stability
  if isstationary(fun) && !isbanded(fun)
    rhsbanded!(RHS, fun, dom)
  end

  # set blocks of constraints
  rhsconstraints!(fitted, gₒ)

  # knock out entries with missing values
  rhsmissings!(RHS, miss)

  nothing
end

# convert RHS into banded matrix
function rhsbanded!(RHS, fun, dom)
  # retrieve total sill
  S = ustrip.(sill(fun))

  # retrieve matrix paramaters
  nobs = nelements(dom)
  nvar = nvariates(fun)
  nfun = nobs * nvar

  @inbounds for j in 1:nvar, i in 1:nfun
    RHS[i, j] = S[mod1(i, nvar), mod1(j, nvar)] - RHS[i, j]
  end

  nothing
end

# knock out entries with missing values
function rhsmissings!(RHS, miss)
  nvar = size(RHS, 2)
  @inbounds for j in 1:nvar, i in miss
    RHS[i, j] = 0
  end

  nothing
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
    Kriging(fun, mean)

Equivalent to [`SimpleKriging`](@ref).

    Kriging(fun)

Equivalent to [`OrdinaryKriging`](@ref).

    Kriging(fun, drifts)

Equivalent to [`UniversalKriging`](@ref).

    Kriging(fun, deg, dim)

Equivalent to [`UniversalKriging`](@ref).

Please check the docstring of corresponding models for more details.
"""
Kriging(f::GeoStatsFunction) = OrdinaryKriging(f)
Kriging(f::GeoStatsFunction, μ::Number) = SimpleKriging(f, μ)
Kriging(f::GeoStatsFunction, μ::AbstractVector{<:Number}) = SimpleKriging(f, μ)
Kriging(f::GeoStatsFunction, deg::Int, dim::Int) = UniversalKriging(f, deg, dim)
Kriging(f::GeoStatsFunction, drifts::AbstractVector) = UniversalKriging(f, drifts)
