# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

function absunits(table)
  cols = Tables.columns(table)
  vars = Tables.columnnames(cols)

  pairs = (var => absunit(Tables.getcolumn(cols, var)) for var in vars)
  (; pairs...) |> Tables.materializer(table)
end

absunit(x) = absunit(nonmissingtype(eltype(x)), x)
absunit(::Type, x) = x
function absunit(::Type{Q}, x) where {Q<:AffineQuantity}
  u = absoluteunit(unit(Q))
  map(v -> uconvert(u, v), x)
end

pointsupport(domain) = PointSet(centroid(domain, i) for i in 1:nelements(domain))
