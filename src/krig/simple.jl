# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    SimpleKriging(γ, μ)

Simple Kriging with variogram model `γ` and constant mean `μ`.

### Notes

* Simple Kriging requires stationary variograms
"""
struct SimpleKriging{G<:Variogram,V} <: KrigingModel
  # input fields
  γ::G
  μ::V

  function SimpleKriging{G,V}(γ, μ) where {G<:Variogram,V}
    @assert isstationary(γ) "Simple Kriging requires stationary variogram"
    new(γ, μ)
  end
end

SimpleKriging(γ, μ) = SimpleKriging{typeof(γ),typeof(μ)}(γ, μ)

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
