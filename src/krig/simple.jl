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

function predictmean(fitted::FittedKriging{<:SimpleKriging}, weights::KrigingWeights, var)
  μ = fitted.model.μ
  d = fitted.state.data
  c = Tables.columns(values(d))
  z = Tables.getcolumn(c, var)
  λ = weights.λ
  y = [zᵢ - μ for zᵢ in z]
  μ + sum(λ .* y)
end
