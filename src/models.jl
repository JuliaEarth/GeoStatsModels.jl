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
    fit(model, data)

Fit geostatistical `model` to geospatial `data` and return a
fitted geostatistical model.
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

# ----------------
# IMPLEMENTATIONS
# ----------------

include("nn.jl")
include("idw.jl")
include("lwr.jl")
include("poly.jl")
include("krig.jl")
