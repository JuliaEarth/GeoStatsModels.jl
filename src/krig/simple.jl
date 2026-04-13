# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    SimpleKriging(fun, mean)

Simple Kriging with geostatistical function `fun` and constant `mean`.

## Examples

```julia
# univariate model with mean 5.0
SimpleKriging(SphericalVariogram(), 5.0)

# multivariate model with mean [5.0, 10.0]
SimpleKriging(I(2) * SphericalVariogram(), [5.0, 10.0])
```

See also [`OrdinaryKriging`](@ref) and [`UniversalKriging`](@ref)
for related Kriging models and the general [`Kriging`](@ref)
constructor that selects the appropriate variant as a function
of the arguments.

### Notes

Simple Kriging requires stationary geostatistical function.
"""
struct SimpleKriging{F<:GeoStatsFunction,M<:AbstractVector} <: KrigingModel
  # input fields
  fun::F
  mean::M

  function SimpleKriging{F,M}(fun, mean) where {F<:GeoStatsFunction,M<:AbstractVector}
    mlen = length(mean)
    nvar = nvariates(fun)
    @assert isstationary(fun) "Simple Kriging requires stationary geostatistical function"
    @assert mlen == nvar || isone(mlen) "length of mean vector must match number of covariates in geostatistical function"
    new(fun, mean)
  end
end

SimpleKriging(fun::F, mean::M) where {F<:GeoStatsFunction,M<:AbstractVector} = SimpleKriging{F,M}(fun, mean)

SimpleKriging(fun, mean) = SimpleKriging(fun, [mean])

scale(model::SimpleKriging, α) = SimpleKriging(GeoStatsFunctions.scale(model.fun, α), model.mean)

nconstraints(::SimpleKriging) = 0

lhsconstraints!(::SimpleKriging, LHS::AbstractMatrix, domain) = nothing

rhsconstraints!(::FittedKriging{<:SimpleKriging}, gₒ) = nothing

function krigmean(fitted::FittedKriging{<:SimpleKriging}, weights::KrigingWeights, vars)
  d = fitted.state.data
  μ = fitted.model.mean
  λ = weights.λ
  k = size(λ, 2)

  cols = Tables.columns(values(d))
  @inbounds map(1:k) do j
    μ[j] + sum(1:k) do p
      λₚ = @view λ[p:k:end, j]
      zₚ = Tables.getcolumn(cols, vars[p])
      sum(i -> λₚ[i] ⦿ (zₚ[i] - μ[p]), eachindex(λₚ, zₚ))
    end
  end
end
