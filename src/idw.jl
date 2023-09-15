# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    IDW(distance=Euclidean(), exponent=1)

The inverse distance weighting model introduced in
the very early days of geostatistics by Shepard 1968.

## References

* Shepard 1968. [A two-dimensional interpolation function
  for irregularly-spaced data](https://dl.acm.org/doi/10.1145/800186.810616)
"""
struct IDW{D,E} <: GeoStatsModel
  distance::D
  exponent::E
end
