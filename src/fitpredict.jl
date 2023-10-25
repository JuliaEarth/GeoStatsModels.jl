# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

function fitpredict(
  model::GeoStatsModel,
  geotable::GeoTable,
  pdomain::Domain;
  neighbors=true,
  # neighbors kwargs
  minneighbors=1,
  maxneighbors=10,
  distance=Euclidean(),
  neighborhood=nothing,
  # other kwargs
  point=true,
  prob=false,
  path=LinearPath()
)
  nobs = nrow(geotable)
  table = values(geotable)
  ddomain = domain(geotable)
  vars = collect(Tables.schema(table).names)

  # adjust data
  data = if point
    pset = PointSet(centroid(ddomain, i) for i in 1:nobs)
    _adjustunits(georef(values(geotable), pset))
  else
    _adjustunits(geotable)
  end

  # prediction order
  inds = traverse(pdomain, path)

  # predict function
  predfun = prob ? predictprob : predict

  pairs = if neighbors
    # fix neighbors limits
    if maxneighbors > nobs || maxneighbors < 1
      @warn "Invalid maximum number of neighbors. Adjusting to $nobs..."
      maxneighbors = nobs
    end

    if minneighbors > maxneighbors || minneighbors < 1
      @warn "Invalid minimum number of neighbors. Adjusting to 1..."
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

    map(vars) do var
      pred = map(inds) do ind
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
        else # missing prediction
          missing
        end
      end

      var => pred
    end
  else
    # fit model to data
    fmodel = fit(model, data)

    map(vars) do var
      pred = map(inds) do ind
        geom = point ? centroid(pdomain, ind) : pdomain[ind]
        predfun(fmodel, var, geom)
      end

      var => pred
    end
  end

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

  vals = Dict(paramdim(dom) => newtab)
  constructor(geotable)(dom, vals)
end

_absunit(x) = _absunit(nonmissingtype(eltype(x)), x)
_absunit(::Type, x) = x
function _absunit(::Type{Q}, x) where {Q<:AffineQuantity}
  u = absoluteunit(unit(Q))
  map(v -> uconvert(u, v), x)
end
