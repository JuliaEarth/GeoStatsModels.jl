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

Fit model to geospatial `data` and return a fitted model.
"""
function fit end

"""
    predict(model, var, uₒ)

Posterior mean of variable `var` at geometry `uₒ`.
"""
function predict end

"""
    predictprob(model, var, uₒ)

Posterior distribution of variable `var` at geometry `uₒ`.
"""
function predictprob end

"""
    status(fitted)

Return the status of the `fitted` model. (e.g. the
factorization of the linear system was successful)
"""
function status end
