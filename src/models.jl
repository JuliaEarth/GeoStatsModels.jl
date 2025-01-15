# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    GeoStatsModel

A geostatistical model that predicts variables over geometries
of a geospatial domain that are in between other geometries with
samples.
"""
abstract type GeoStatsModel end

"""
    fit(model, data)

Fit geostatistical `model` to geospatial `data` and return a
fitted geostatistical model.
"""
function fit end

"""
    predict(model, vars, gₒ)

Predict one or multiple variables `vars` at geometry `gₒ` with
given geostatistical `model`.
"""
function predict end

"""
    predictprob(model, vars, gₒ)

Predict distribution of one or multiple variables `vars` at
geometry `gₒ` with given geostatistical `model`.
"""
function predictprob end

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
