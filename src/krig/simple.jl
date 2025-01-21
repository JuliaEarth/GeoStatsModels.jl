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

nconstraints(::SimpleKriging, ::Int) = 0

lhsconstraints!(::SimpleKriging, LHS::AbstractMatrix, nvar::Int, domain) = nothing

rhsconstraints!(::FittedKriging{<:SimpleKriging}, gₒ) = nothing

function krigmean(fitted::FittedKriging{<:SimpleKriging}, weights::KrigingWeights, vars)
  d = fitted.state.data
  μ = fitted.model.mean
  λ = weights.λ
  k = length(vars)

  @assert size(λ, 2) == k "invalid number of variables for Kriging model"

  cols = Tables.columns(values(d))
  @inbounds ntuple(k) do j
    sum(1:k) do p
      λₚ = @view λ[p:k:end, j]
      zₚ = Tables.getcolumn(cols, vars[p])
      μ[p] + sum(i -> λₚ[i] * (zₚ[i] - μ[p]), eachindex(λₚ, zₚ))
    end
  end
end
