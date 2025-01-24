# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    GeoStatsModel

A geostatistical model that predicts variables over geometries
of a geospatial domain near other geometries with samples.
"""
abstract type GeoStatsModel end

"""
    fit(model, geotable)

Fit geostatistical `model` to `geotable` and return a fitted
geostatistical model.
"""
function fit end

"""
    FittedGeoStatsModel

A fitted geostatistical model obtained with the [`fit`](@ref) function
on a [`GeoStatsModel`](@ref).
"""
abstract type FittedGeoStatsModel end

"""
    predict(model, vars, gₒ)

Predict one or multiple variables `vars` at geometry `gₒ` with
given geostatistical `model`.
"""
predict(model::FittedGeoStatsModel, var::AbstractString, gₒ) = predict(model, Symbol(var), gₒ)
predict(model::FittedGeoStatsModel, vars, gₒ) = [predict(model, var, gₒ) for var in vars]

"""
    predictprob(model, vars, gₒ)

Predict distribution of one or multiple variables `vars` at
geometry `gₒ` with given geostatistical `model`.
"""
predictprob(model::FittedGeoStatsModel, var::AbstractString, gₒ) = predictprob(model, Symbol(var), gₒ)
predictprob(model::FittedGeoStatsModel, vars, gₒ) = [predictprob(model, var, gₒ) for var in vars]

"""
    status(fitted)

Return the status of the `fitted` geostatistical model.
(e.g. the factorization of the linear system was successful)
"""
function status end

"""
    fitpredict(model, geotable, domain; [parameters])

Fit geostatistical `model` to `geotable` and predict all
variables on `domain` using a set of optional parameters.

## Parameters

* `path`         - Path over the domain (default to `LinearPath()`)
* `point`        - Perform interpolation on point support (default to `true`)
* `prob`         - Perform probabilistic interpolation (default to `false`)
* `neighbors`    - Whether or not to use neighborhood (default to `true`)
* `minneighbors` - Minimum number of neighbors (default to `1`)
* `maxneighbors` - Maximum number of neighbors (default to `10`)
* `neighborhood` - Search neighborhood (default to `nothing`)
* `distance`     - Distance to find nearest neighbors (default to `Euclidean()`)
"""
function fitpredict(
  model::GeoStatsModel,
  gtb::AbstractGeoTable,
  dom::Domain;
  path=LinearPath(),
  point=true,
  prob=false,
  neighbors=true,
  minneighbors=1,
  maxneighbors=10,
  neighborhood=nothing,
  distance=Euclidean()
)
  # point or volume support
  sdom = point ? pointsupport(domain(gtb)) : domain(gtb)

  # adjusted geotable
  gtb′ = georef(values(gtb), sdom)

  if neighbors
    fitpredictneigh(model, gtb′, dom, path, point, prob, minneighbors, maxneighbors, neighborhood, distance)
  else
    fitpredictfull(model, gtb′, dom, path, point, prob)
  end
end

function fitpredictneigh(
  model,
  gtb,
  dom,
  path,
  point,
  prob,
  minneighbors,
  maxneighbors,
  neighborhood,
  distance
)
  # fix neighbors limits
  nobs = nrow(gtb)
  if maxneighbors > nobs || maxneighbors < 1
    maxneighbors = nobs
  end
  if minneighbors > maxneighbors || minneighbors < 1
    minneighbors = 1
  end

  # determine bounded search method
  searcher = if isnothing(neighborhood)
    # nearest neighbor search with a metric
    KNearestSearch(domain(gtb), maxneighbors; metric=distance)
  else
    # neighbor search with ball neighborhood
    KBallSearch(domain(gtb), maxneighbors, neighborhood)
  end

  # pre-allocate memory for neighbors
  neighbors = Vector{Int}(undef, maxneighbors)

  # traverse domain with given path
  inds = traverse(dom, path)

  # prediction function
  predfun = prob ? predictprob : predict

  # predict variables
  cols = Tables.columns(values(gtb))
  vars = Tables.columnnames(cols)
  pred = @inbounds map(inds) do ind
    # centroid of estimation
    center = centroid(dom, ind)

    # find neighbors with data
    nneigh = search!(neighbors, center, searcher)

    # predict if enough neighbors
    if nneigh ≥ minneighbors
      # final set of neighbors
      ninds = view(neighbors, 1:nneigh)

      # view neighborhood with data
      samples = view(gtb, ninds)

      # fit model to samples
      fmodel = fit(model, samples)

      # save prediction
      geom = point ? center : dom[ind]
      vals = predfun(fmodel, vars, geom)
    else
      # missing prediction
      vals = fill(missing, length(vars))
    end
    (; zip(vars, vals)...)
  end

  # convert to original table type
  predtab = pred |> Tables.materializer(values(gtb))

  georef(predtab, dom)
end

function fitpredictfull(model, gtb, dom, path, point, prob)
  # traverse domain with given path
  inds = traverse(dom, path)

  # prediction function
  predfun = prob ? predictprob : predict

  # fit model to data
  fmodel = fit(model, gtb)

  # predict variables
  cols = Tables.columns(values(gtb))
  vars = Tables.columnnames(cols)
  pred = @inbounds map(inds) do ind
    geom = point ? centroid(dom, ind) : dom[ind]
    vals = predfun(fmodel, vars, geom)
    (; zip(vars, vals)...)
  end

  # convert to original table type
  predtab = pred |> Tables.materializer(values(gtb))

  georef(predtab, dom)
end

# ----------------
# IMPLEMENTATIONS
# ----------------

include("nn.jl")
include("idw.jl")
include("lwr.jl")
include("poly.jl")
include("krig.jl")

# -----------------
# HELPER FUNCTIONS
# -----------------

pointsupport(dom) = PointSet(centroid(dom, i) for i in 1:nelements(dom))
