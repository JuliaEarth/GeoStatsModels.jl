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

nconstraints(::SimpleKriging, ::Int) = 0

set_constraints_lhs!(::SimpleKriging, LHS::AbstractMatrix, nvar::Int, domain) = nothing

set_constraints_rhs!(::FittedKriging{<:SimpleKriging}, gₒ) = nothing

function krigmean(fitted::FittedKriging{<:SimpleKriging}, weights::KrigingWeights, vars)
  d = fitted.state.data
  k = fitted.state.nvar
  μ = fitted.model.μ
  λ = weights.λ

  cols = Tables.columns(values(d))
  @inbounds ntuple(k) do j
    sum(1:k) do p
      λₚ = @view λ[p:k:end, j]
      zₚ = Tables.getcolumn(cols, vars[p])
      μ[p] + sum(i -> λₚ[i] * (zₚ[i] - μ[p]), eachindex(λₚ, zₚ))
    end
  end
end
