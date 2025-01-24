# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    fitpredict(model, data, domain; [parameters])

Fit geostatistical `model` to `data` and predict on `domain`
using a set of optional parameters.

## Parameters

* `neighbors`    - Whether or not to use neighborhood (default to `true`)
* `minneighbors` - Minimum number of neighbors (default to `1`)
* `maxneighbors` - Maximum number of neighbors (default to `10`)
* `neighborhood` - Search neighborhood (default to `nothing`)
* `distance`     - Distance to find nearest neighbors (default to `Euclidean()`)
* `path`         - Path over the domain (default to `LinearPath()`)
* `point`        - Perform interpolation on point support (default to `true`)
* `prob`         - Perform probabilistic interpolation (default to `false`)
"""
function fitpredict(
  model::GeoStatsModel,
  geotable::AbstractGeoTable,
  pdomain::Domain;
  neighbors=true,
  minneighbors=1,
  maxneighbors=10,
  neighborhood=nothing,
  distance=Euclidean(),
  path=LinearPath(),
  point=true,
  prob=false
)
  if neighbors
    fitpredictneigh(model, geotable, pdomain, minneighbors, maxneighbors, neighborhood, distance, path, point, prob)
  else
    fitpredictall(model, geotable, pdomain, path, point, prob)
  end
end

function fitpredictall(model, geotable, pdomain, path, point, prob)
  table = values(geotable)
  ddomain = domain(geotable)
  vars = Tables.schema(table).names

  # adjust data
  data = if point
    pset = PointSet(centroid(ddomain, i) for i in 1:nelements(ddomain))
    _adjustunits(georef(values(geotable), pset))
  else
    _adjustunits(geotable)
  end

  # prediction order
  inds = traverse(pdomain, path)

  # predict function
  predfun = prob ? predictprob : predict

  # fit model to data
  fmodel = fit(model, data)

  # predict variable values
  function pred(var)
    map(inds) do ind
      geom = point ? centroid(pdomain, ind) : pdomain[ind]
      predfun(fmodel, var, geom)
    end
  end

  pairs = (var => pred(var) for var in vars)
  newtab = (; pairs...) |> Tables.materializer(table)
  georef(newtab, pdomain)
end

function fitpredictneigh(
  model,
  geotable,
  pdomain,
  minneighbors,
  maxneighbors,
  neighborhood,
  distance,
  path,
  point,
  prob
)
  table = values(geotable)
  ddomain = domain(geotable)
  vars = Tables.schema(table).names

  # adjust data
  data = if point
    pset = PointSet(centroid(ddomain, i) for i in 1:nelements(ddomain))
    _adjustunits(georef(values(geotable), pset))
  else
    _adjustunits(geotable)
  end

  # fix neighbors limits
  nobs = nrow(data)
  if maxneighbors > nobs || maxneighbors < 1
    maxneighbors = nobs
  end
  if minneighbors > maxneighbors || minneighbors < 1
    minneighbors = 1
  end

  # determine bounded search method
  searcher = if isnothing(neighborhood)
    # nearest neighbor search with a metric
    KNearestSearch(ddomain, maxneighbors; metric=distance)
  else
    # neighbor search with ball neighborhood
    KBallSearch(ddomain, maxneighbors, neighborhood)
  end

  # pre-allocate memory for neighbors
  neighbors = Vector{Int}(undef, maxneighbors)

  # prediction order
  inds = traverse(pdomain, path)

  # predict function
  predfun = prob ? predictprob : predict

  # predict variable values
  function pred(var)
    map(inds) do ind
      # centroid of estimation
      center = centroid(pdomain, ind)

      # find neighbors with data
      nneigh = search!(neighbors, center, searcher)

      # predict if enough neighbors
      if nneigh ≥ minneighbors
        # final set of neighbors
        ninds = view(neighbors, 1:nneigh)

        # view neighborhood with data
        samples = view(data, ninds)

        # fit model to samples
        fmodel = fit(model, samples)

        # save prediction
        geom = point ? center : pdomain[ind]
        predfun(fmodel, var, geom)
      else
        # missing prediction
        missing
      end
    end
  end

  pairs = (var => pred(var) for var in vars)
  newtab = (; pairs...) |> Tables.materializer(table)
  georef(newtab, pdomain)
end

#--------------
# ADJUST UNITS
#--------------

function _adjustunits(geotable::AbstractGeoTable)
  dom = domain(geotable)
  tab = values(geotable)
  cols = Tables.columns(tab)
  vars = Tables.columnnames(cols)

  pairs = (var => _absunit(Tables.getcolumn(cols, var)) for var in vars)
  newtab = (; pairs...) |> Tables.materializer(tab)

  georef(newtab, dom)
end

_absunit(x) = _absunit(nonmissingtype(eltype(x)), x)
_absunit(::Type, x) = x
function _absunit(::Type{Q}, x) where {Q<:AffineQuantity}
  u = absoluteunit(unit(Q))
  map(v -> uconvert(u, v), x)
end
