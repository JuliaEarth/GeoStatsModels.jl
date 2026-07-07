using BenchmarkTools
using GeoStatsModels
using GeoStatsFunctions
using GeoTables
using Meshes

# auxiliary variables
table1 = (; z=[1.0, 2.0, 3.0])
table2 = (; z=[missing, 2.0, 3.0])
coords = [(25.0, 25.0), (50.0, 75.0), (75.0, 50.0)]
data1 = georef(table1, coords)
data2 = georef(table2, coords)
igrid = CartesianGrid(100, 100)
model = Kriging(GaussianVariogram(range=35.0))

# initialize benchmark suite
const SUITE = BenchmarkGroup()

# --------
# KRIGING
# --------

SUITE["kriging"] = BenchmarkGroup()

SUITE["kriging"]["full"] = @benchmarkable GeoStatsModels.fitpredict($model, $data1, $igrid, neighbors=true)
SUITE["kriging"]["miss"] = @benchmarkable GeoStatsModels.fitpredict($model, $data2, $igrid, neighbors=true)
