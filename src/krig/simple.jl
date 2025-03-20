# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    SimpleKriging(fun, mean)

Simple Kriging with geostatistical function `fun` and constant `mean`.

### Notes

* Simple Kriging requires stationary geostatistical function
"""
struct SimpleKriging{F<:GeoStatsFunction,M<:AbstractVector} <: KrigingModel
  # input fields
  fun::F
  mean::M

  function SimpleKriging{F,M}(fun, mean) where {F<:GeoStatsFunction,M<:AbstractVector}
    @assert isstationary(fun) "Simple Kriging requires stationary geostatistical function"
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
