# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    LWR(distance=Euclidean(), weightfun=h -> exp(-3 * h^2))

The locally weighted regression (a.k.a. LOESS) model introduced by
Cleveland 1979. It is the most natural generalization of [`IDW`](@ref)
in which one is allowed to use a custom weight function instead of
distance-based weights.

## References

* Stone 1977. *Consistent non-parametric regression.*
* Cleveland 1979. *Robust locally weighted regression and smoothing scatterplots.*
* Cleveland & Grosse 1991. *Computational methods for local regression.*

### Notes

* In the current implementation, the variance is computed assuming normal residuals. 
"""
struct LWR{D,F} <: GeoStatsModel
  distance::D
  weightfun::F
end

