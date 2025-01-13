# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    SimpleKriging(f, μ)

Simple Kriging with geostatistical function `f` and constant mean `μ`.

### Notes

* Simple Kriging requires stationary geostatistical function
"""
struct SimpleKriging{F<:GeoStatsFunction,M} <: KrigingModel
  # input fields
  f::F
  μ::M

  function SimpleKriging{F,M}(f, μ) where {F<:GeoStatsFunction,M}
    @assert isstationary(f) "Simple Kriging requires stationary geostatistical function"
    new(f, μ)
  end
end

SimpleKriging(f, μ) = SimpleKriging{typeof(f),typeof(μ)}(f, μ)

nconstraints(::SimpleKriging) = 0

set_constraints_lhs!(::SimpleKriging, LHS::AbstractMatrix, domain) = nothing

set_constraints_rhs!(::FittedKriging{<:SimpleKriging}, gₒ) = nothing

function krigmean(fitted::FittedKriging{<:SimpleKriging}, weights::KrigingWeights, vars)
  d = fitted.state.data
  k = fitted.state.nvar
  μ = fitted.model.μ
  λ = weights.λ

  cols = Tables.columns(values(d))
  @inbounds ntuple(k) do j
    λⱼ = @view λ[j:k:end, j]
    zⱼ = Tables.getcolumn(cols, vars[j])
    μ[j] + sum(i -> λⱼ[i] * (zⱼ[i] - μ[j]), eachindex(λⱼ, zⱼ))
  end
end
