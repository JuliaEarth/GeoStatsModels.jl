@testset "Kriging" begin
  rng = StableRNG(2024)

  tol = 10 * eps(Float64)

  SK = GeoStatsModels.SimpleKriging
  OK = GeoStatsModels.OrdinaryKriging
  UK = GeoStatsModels.UniversalKriging
  DK = GeoStatsModels.UniversalKriging

  @testset "Basics" begin
    dim = 3
    nobs = 10
    cmat = 10 * rand(rng, dim, nobs)
    pset = PointSet(Tuple.(eachcol(cmat)))
    data = georef((z=rand(rng, nobs),), pset)

    γ = GaussianVariogram(sill=1.0, range=1.0, nugget=0.0)
    simkrig = SK(γ, mean(data.z))
    ordkrig = OK(γ)
    unikrig = UK(γ, 1, 3)
    drikrig = DK(γ, [x -> 1.0])

    sk = GeoStatsModels.fit(simkrig, data)
    ok = GeoStatsModels.fit(ordkrig, data)
    uk = GeoStatsModels.fit(unikrig, data)
    dk = GeoStatsModels.fit(drikrig, data)

    # Kriging is an interpolator
    for j in 1:nobs
      skdist = GeoStatsModels.predictprob(sk, :z, pset[j])
      okdist = GeoStatsModels.predictprob(ok, :z, pset[j])
      ukdist = GeoStatsModels.predictprob(uk, :z, pset[j])
      dkdist = GeoStatsModels.predictprob(dk, :z, pset[j])

      # mean checks
      @test mean(skdist) ≈ data.z[j]
      @test mean(okdist) ≈ data.z[j]
      @test mean(ukdist) ≈ data.z[j]
      @test mean(dkdist) ≈ data.z[j]

      # variance checks
      @test var(skdist) ≥ 0
      @test var(okdist) ≥ 0
      @test var(ukdist) ≥ 0
      @test var(dkdist) ≥ 0
      @test var(skdist) ≤ var(okdist) + tol
    end

    # save results on a particular location pₒ
    pₒ = rand(rng, Point)
    skdist = GeoStatsModels.predictprob(sk, :z, pₒ)
    okdist = GeoStatsModels.predictprob(ok, :z, pₒ)
    ukdist = GeoStatsModels.predictprob(uk, :z, pₒ)
    dkdist = GeoStatsModels.predictprob(dk, :z, pₒ)

    # Kriging is translation-invariant
    h = to(rand(rng, Point))
    pset_h = PointSet([pset[i] + h for i in 1:nelements(pset)])
    data_h = georef((z=data.z,), pset_h)
    sk_h = GeoStatsModels.fit(SK(γ, mean(data_h.z)), data_h)
    ok_h = GeoStatsModels.fit(OK(γ), data_h)
    uk_h = GeoStatsModels.fit(UK(γ, 1, 3), data_h)
    dk_h = GeoStatsModels.fit(DK(γ, [x -> 1.0]), data_h)
    skdist_h = GeoStatsModels.predictprob(sk_h, :z, pₒ + h)
    okdist_h = GeoStatsModels.predictprob(ok_h, :z, pₒ + h)
    ukdist_h = GeoStatsModels.predictprob(uk_h, :z, pₒ + h)
    dkdist_h = GeoStatsModels.predictprob(dk_h, :z, pₒ + h)
    @test mean(skdist_h) ≈ mean(skdist)
    @test var(skdist_h) ≈ var(skdist)
    @test mean(okdist_h) ≈ mean(okdist)
    @test var(okdist_h) ≈ var(okdist)
    @test mean(ukdist_h) ≈ mean(ukdist)
    @test var(ukdist_h) ≈ var(ukdist)
    @test mean(dkdist_h) ≈ mean(dkdist)
    @test var(dkdist_h) ≈ var(dkdist)

    # Kriging mean is invariant under covariance scaling
    # Kriging variance is multiplied by the same factor
    α = 2.0
    γ_α = GaussianVariogram(sill=α, range=1.0, nugget=0.0)
    sk_α = GeoStatsModels.fit(SK(γ_α, mean(data.z)), data)
    ok_α = GeoStatsModels.fit(OK(γ_α), data)
    uk_α = GeoStatsModels.fit(UK(γ_α, 1, 3), data)
    dk_α = GeoStatsModels.fit(DK(γ_α, [x -> 1.0]), data)
    skdist_α = GeoStatsModels.predictprob(sk_α, :z, pₒ)
    okdist_α = GeoStatsModels.predictprob(ok_α, :z, pₒ)
    ukdist_α = GeoStatsModels.predictprob(uk_α, :z, pₒ)
    dkdist_α = GeoStatsModels.predictprob(dk_α, :z, pₒ)
    @test mean(skdist_α) ≈ mean(skdist)
    @test var(skdist_α) ≈ α * var(skdist)
    @test mean(okdist_α) ≈ mean(okdist)
    @test var(okdist_α) ≈ α * var(okdist)
    @test mean(ukdist_α) ≈ mean(ukdist)
    @test var(ukdist_α) ≈ α * var(ukdist)
    @test mean(dkdist_α) ≈ mean(dkdist)
    @test var(dkdist_α) ≈ α * var(dkdist)

    # Kriging variance is a function of data configuration, not data values
    δ = rand(rng, nobs)
    data_δ = georef((z=data.z .+ δ,), pset)
    sk_δ = GeoStatsModels.fit(SK(γ, mean(data_δ.z)), data_δ)
    ok_δ = GeoStatsModels.fit(OK(γ), data_δ)
    uk_δ = GeoStatsModels.fit(UK(γ, 1, 3), data_δ)
    dk_δ = GeoStatsModels.fit(DK(γ, [x -> 1.0]), data_δ)
    skdist_δ = GeoStatsModels.predictprob(sk_δ, :z, pₒ)
    okdist_δ = GeoStatsModels.predictprob(ok_δ, :z, pₒ)
    ukdist_δ = GeoStatsModels.predictprob(uk_δ, :z, pₒ)
    dkdist_δ = GeoStatsModels.predictprob(dk_δ, :z, pₒ)
    @test var(skdist_δ) ≈ var(skdist)
    @test var(okdist_δ) ≈ var(okdist)
    @test var(ukdist_δ) ≈ var(ukdist)
    @test var(dkdist_δ) ≈ var(dkdist)

    # Ordinary Kriging ≡ Universal Kriging with 0th degree drift
    uk_0 = GeoStatsModels.fit(UK(γ, 0, 3), data)
    okdist = GeoStatsModels.predictprob(ok, :z, pₒ)
    ukdist_0 = GeoStatsModels.predictprob(uk_0, :z, pₒ)
    @test mean(okdist) ≈ mean(ukdist_0)
    @test var(okdist) ≈ var(ukdist_0)

    # Ordinary Kriging ≡ Kriging with constant external drift
    dk_c = GeoStatsModels.fit(DK(γ, [x -> 1.0]), data)
    okdist = GeoStatsModels.predictprob(ok, :z, pₒ)
    dkdist_c = GeoStatsModels.predictprob(dk_c, :z, pₒ)
    @test mean(okdist) ≈ mean(dkdist_c)
    @test var(okdist) ≈ var(dkdist_c)

    # latlon coordinates
    pts = Point.([LatLon(0, 0), LatLon(0, 1), LatLon(1, 0)])
    nobs = length(pts)
    data = georef((; z=rand(rng, nobs)), pts)
    sk = GeoStatsModels.fit(SK(γ, mean(data.z)), data)
    ok = GeoStatsModels.fit(OK(γ), data)
    uk = GeoStatsModels.fit(UK(γ, 1, 2), data)
    dk = GeoStatsModels.fit(DK(γ, [x -> 1.0]), data)
    for i in 1:nobs
      skdist = GeoStatsModels.predictprob(sk, :z, pts[i])
      okdist = GeoStatsModels.predictprob(ok, :z, pts[i])
      ukdist = GeoStatsModels.predictprob(uk, :z, pts[i])
      dkdist = GeoStatsModels.predictprob(dk, :z, pts[i])

      # mean checks
      @test mean(skdist) ≈ data.z[i]
      @test mean(okdist) ≈ data.z[i]
      @test mean(ukdist) ≈ data.z[i]
      @test mean(dkdist) ≈ data.z[i]
    end
  end

  # non-stationary variograms are allowed
  @testset "Stationarity" begin
    dim = 3
    nobs = 10
    cmat = 10 * rand(rng, dim, nobs)
    pset = PointSet(Tuple.(eachcol(cmat)))
    data = georef((z=rand(rng, nobs),), pset)

    γ_ns = PowerVariogram()
    ok_ns = GeoStatsModels.fit(OK(γ_ns), data)
    uk_ns = GeoStatsModels.fit(UK(γ_ns, 1, 3), data)
    dk_ns = GeoStatsModels.fit(DK(γ_ns, [x -> 1.0]), data)
    for j in 1:nobs
      okdist_ns = GeoStatsModels.predictprob(ok_ns, :z, pset[j])
      ukdist_ns = GeoStatsModels.predictprob(uk_ns, :z, pset[j])
      dkdist_ns = GeoStatsModels.predictprob(dk_ns, :z, pset[j])

      # mean checks
      @test mean(okdist_ns) ≈ data.z[j]
      @test mean(ukdist_ns) ≈ data.z[j]
      @test mean(dkdist_ns) ≈ data.z[j]

      # variance checks
      @test var(okdist_ns) ≥ 0
      @test var(ukdist_ns) ≥ 0
      @test var(dkdist_ns) ≥ 0
    end
  end

  # floating point checks
  @testset "Floats" begin
    dim = 3
    nobs = 10
    X_f = rand(rng, Float32, dim, nobs)
    z_f = rand(rng, Float32, nobs)
    X_d = Float64.(X_f)
    z_d = Float64.(z_f)
    pset_f = PointSet(Tuple.(eachcol(X_f)))
    data_f = georef((z=z_f,), pset_f)
    pset_d = PointSet(Tuple.(eachcol(X_d)))
    data_d = georef((z=z_d,), pset_d)
    coords_f = ntuple(i -> rand(rng, Float32), dim)
    coords_d = Float64.(coords_f)
    pₒ_f = Point(coords_f)
    pₒ_d = Point(coords_d)
    γ_f = GaussianVariogram(sill=1.0f0, range=1.0f0, nugget=0.0f0)
    sk_f = GeoStatsModels.fit(SK(γ_f, mean(data_f.z)), data_f)
    ok_f = GeoStatsModels.fit(OK(γ_f), data_f)
    uk_f = GeoStatsModels.fit(UK(γ_f, 1, 3), data_f)
    dk_f = GeoStatsModels.fit(DK(γ_f, [x -> 1.0f0]), data_f)
    γ_d = GaussianVariogram(sill=1.0, range=1.0, nugget=0.0)
    sk_d = GeoStatsModels.fit(SK(γ_d, mean(data_d.z)), data_d)
    ok_d = GeoStatsModels.fit(OK(γ_d), data_d)
    uk_d = GeoStatsModels.fit(UK(γ_d, 1, 3), data_d)
    dk_d = GeoStatsModels.fit(DK(γ_d, [x -> 1.0]), data_d)
    skdist_f = GeoStatsModels.predictprob(sk_f, :z, pₒ_f)
    okdist_f = GeoStatsModels.predictprob(ok_f, :z, pₒ_f)
    ukdist_f = GeoStatsModels.predictprob(uk_f, :z, pₒ_f)
    dkdist_f = GeoStatsModels.predictprob(dk_f, :z, pₒ_f)
    skdist_d = GeoStatsModels.predictprob(sk_d, :z, pₒ_d)
    okdist_d = GeoStatsModels.predictprob(ok_d, :z, pₒ_d)
    ukdist_d = GeoStatsModels.predictprob(uk_d, :z, pₒ_d)
    dkdist_d = GeoStatsModels.predictprob(dk_d, :z, pₒ_d)
    @test isapprox(mean(skdist_f), mean(skdist_d), atol=1e-4)
    @test isapprox(var(skdist_f), var(skdist_d), atol=1e-4)
    @test isapprox(mean(okdist_f), mean(okdist_d), atol=1e-4)
    @test isapprox(var(okdist_f), var(okdist_d), atol=1e-4)
    @test isapprox(mean(ukdist_f), mean(ukdist_d), atol=1e-4)
    @test isapprox(var(ukdist_f), var(ukdist_d), atol=1e-4)
    @test isapprox(mean(dkdist_f), mean(dkdist_d), atol=1e-4)
    @test isapprox(var(dkdist_f), var(dkdist_d), atol=1e-4)
  end

  # change of support checks
  @testset "Support" begin
    dim = 2
    nobs = 10
    cmat = 10 * rand(rng, dim, nobs)
    pset = PointSet(Tuple.(eachcol(cmat)))
    data = georef((z=rand(rng, nobs),), pset)

    γ = GaussianVariogram(sill=1.0, range=1.0, nugget=0.0)
    sk = GeoStatsModels.fit(SK(γ, mean(data.z)), data)
    ok = GeoStatsModels.fit(OK(γ), data)
    uk = GeoStatsModels.fit(UK(γ, 1, dim), data)
    dk = GeoStatsModels.fit(DK(γ, [x -> 1.0]), data)

    # predict on a quadrangle
    gₒ = Quadrangle((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    skdist = GeoStatsModels.predictprob(sk, :z, gₒ)
    okdist = GeoStatsModels.predictprob(ok, :z, gₒ)
    ukdist = GeoStatsModels.predictprob(uk, :z, gₒ)
    dkdist = GeoStatsModels.predictprob(dk, :z, gₒ)

    # variance checks
    @test var(skdist) ≥ 0
    @test var(okdist) ≥ 0
    @test var(ukdist) ≥ 0
    @test var(dkdist) ≥ 0
    @test var(skdist) ≤ var(okdist) + tol
  end

  @testset "CoDa" begin
    dim = 2
    nobs = 10
    cmat = 10 * rand(rng, dim, nobs)
    pset = PointSet(Tuple.(eachcol(cmat)))
    table = (z=rand(rng, Composition{3}, nobs),)
    data = georef(table, pset)

    # basic models
    γ = GaussianVariogram(sill=1.0, range=1.0, nugget=0.0)
    sk = GeoStatsModels.fit(SK(γ, mean(data.z)), data)
    ok = GeoStatsModels.fit(OK(γ), data)
    uk = GeoStatsModels.fit(UK(γ, 1, dim), data)
    dk = GeoStatsModels.fit(DK(γ, [x -> 1.0]), data)

    # prediction on a quadrangle
    gₒ = Quadrangle((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    skmean = GeoStatsModels.predict(sk, :z, gₒ)
    okmean = GeoStatsModels.predict(ok, :z, gₒ)
    ukmean = GeoStatsModels.predict(uk, :z, gₒ)
    dkmean = GeoStatsModels.predict(dk, :z, gₒ)

    # type tests
    @test skmean isa Composition
    @test okmean isa Composition
    @test ukmean isa Composition
    @test dkmean isa Composition
  end

  @testset "Unitiful" begin
    dim = 3
    nobs = 10
    cmat = 10 * rand(rng, dim, nobs)
    pset = PointSet(Tuple.(eachcol(cmat)))
    data = georef((z=rand(rng, nobs) * u"K",), pset)

    γ = GaussianVariogram(sill=1.0u"K^2")
    sk = GeoStatsModels.fit(SK(γ, mean(data.z)), data)
    ok = GeoStatsModels.fit(OK(γ), data)
    uk = GeoStatsModels.fit(UK(γ, 1, dim), data)
    dk = GeoStatsModels.fit(DK(γ, [x -> 1.0]), data)
    for _k in [sk, ok, uk, dk]
      w = GeoStatsModels.weights(_k, Point(0, 0, 0))
      μ = GeoStatsModels.predictmean(_k, w, :z)
      σ² = GeoStatsModels.predictvar(_k, w)
      @test unit(μ) == u"K"
      @test unit(σ²) == u"K^2"
    end
  end

  @testset begin "Kriging"
    γ = GaussianVariogram()
    @test Kriging(γ) isa OK
    @test Kriging(γ, 0.0) isa SK
    @test Kriging(γ, 1, 2) isa UK
    @test Kriging(γ, [x -> 1]) isa DK
  end
end
