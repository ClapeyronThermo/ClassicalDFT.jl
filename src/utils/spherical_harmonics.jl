using SpecialFunctions: besselix

"""
    gauss_legendre_nodes(n, ::Type{FP}=Float64) -> (x::Vector{FP}, w::Vector{FP})

Nodes and weights for `n`-point Gauss-Legendre quadrature on `[-1,1]`, via Newton's
method on the standard 3-term Legendre recurrence (seeded with the usual
Chebyshev-node initial guess). Exact for polynomials up to degree `2n-1`. Returned in
ascending order of `x`.
"""
function gauss_legendre_nodes(n::Int, ::Type{FP}=Float64) where FP<:AbstractFloat
    x = zeros(FP, n)
    w = zeros(FP, n)
    for i in 1:n
        xi = cos(FP(pi) * (i - FP(0.25)) / (n + FP(0.5)))
        p0, p1, dp = one(FP), xi, one(FP)
        for _ in 1:100
            p0, p1 = one(FP), xi
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * xi * p1 - (k - 1) * p0) / k
            end
            dp = n * (xi * p1 - p0) / (xi^2 - one(FP))
            δ = p1 / dp
            xi -= δ
            abs(δ) < 10 * eps(FP) && break
        end
        x[i] = xi
        w[i] = 2 / ((1 - xi^2) * dp^2)
    end
    perm = sortperm(x)
    return x[perm], w[perm]
end

"""
    assoc_legendre_table(x::AbstractVector, L_max::Int) -> Vector{Matrix}

For each order `m = 0:L_max`, a matrix `P[m+1]` of shape `(length(x), L_max-m+1)` with
`P[m+1][j, l-m+1] = P_l^m(x[j])` for `l = m:L_max` (Condon-Shortley phase included, the
standard physics convention), via the usual seed-plus-3-term-recurrence algorithm:
`P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}`, `P_{m+1}^m(x) = x(2m+1)P_m^m(x)`,
`(l-m)P_l^m(x) = x(2l-1)P_{l-1}^m(x) - (l+m-1)P_{l-2}^m(x)` for `l ≥ m+2`.
"""
function assoc_legendre_table(x::AbstractVector{FP}, L_max::Int) where FP<:AbstractFloat
    n = length(x)
    P = [zeros(FP, n, L_max - m + 1) for m in 0:L_max]
    for j in 1:n
        xj = x[j]
        somx2 = sqrt(max(1 - xj^2, zero(FP)))
        for m in 0:L_max
            pmm = one(FP)
            fact = one(FP)
            for _ in 1:m
                pmm *= -fact * somx2
                fact += 2
            end
            P[m+1][j, 1] = pmm
            if L_max > m
                pmmp1 = xj * (2m + 1) * pmm
                P[m+1][j, 2] = pmmp1
                pprev2, pprev1 = pmm, pmmp1
                for l in (m+2):L_max
                    pll = (xj * (2l - 1) * pprev1 - (l + m - 1) * pprev2) / (l - m)
                    P[m+1][j, l - m + 1] = pll
                    pprev2, pprev1 = pprev1, pll
                end
            end
        end
    end
    return P
end

# (l-m)!/(l+m)! computed as a bounded product (2m terms, all ≤ 1) to avoid
# overflow/underflow from evaluating the two factorials separately.
function _legendre_factorial_ratio(l::Int, m::Int, ::Type{FP}) where FP<:AbstractFloat
    ratio = one(FP)
    for k in (l - m + 1):(l + m)
        ratio /= FP(k)
    end
    return ratio
end

"""
    SHTPlan{FP}

Real-spherical-harmonic transform plan on a Gauss-Legendre-in-`cosθ` ×
equispaced-in-`φ` orientation quadrature grid, truncated at spherical-harmonic degree
`L_max`. This is the orientation-space analogue of the spatial FFT plans
`DiscreteGaussianChainPropagator` builds: the grid/transform machinery a `WLCPropagator`
needs to apply a bending kernel that is diagonal in spherical-harmonic degree `l` (see
`bending_eigenvalues`).

# Fields
- `L_max`: spherical-harmonic truncation degree.
- `n_theta = L_max+1`, `n_phi = 2L_max+1`: orientation quadrature grid size
  (`n_orient = n_theta*n_phi`). `n_phi` is exactly `2L_max+1` (not just `≥`) since a
  real signal's `rfft` along `φ` then has exactly `L_max+1` frequency bins — one per
  spherical-harmonic order `m = 0:L_max` — with no leftover/aliased frequencies to
  truncate.
- `cosθ`, `gl_weight`: the `n_theta` Gauss-Legendre nodes/weights on `[-1,1]`.
- `φ`: the `n_phi` equispaced azimuthal grid points.
- `u_nodes`: `(3, n_orient)` Cartesian unit vectors for every orientation grid node,
  `θ`-fastest / `φ`-slowest (`u_nodes[:, jθ + (jφ-1)*n_theta]`) — this ordering matches
  `reshape`'s column-major convention for splitting a flat `n_orient` axis into
  `(n_theta, n_phi)` in `sht_forward!`/`sht_inverse!`.
- `quad_weight`: length-`n_orient` S² quadrature weights (`gl_weight[jθ] * 2π/n_phi`,
  same node ordering as `u_nodes`; `sum(quad_weight) ≈ 4π`).
- `Plm_fwd`/`Plm_inv`: per-order (`m=0:L_max`) dense associated-Legendre operator
  matrices, shape `(n_theta, L_max-m+1)`, with the real-spherical-harmonic
  normalization and (for `Plm_fwd`) the `2π` azimuthal-integral factor and
  Gauss-Legendre quadrature weight baked in — see `sht_forward!`/`sht_inverse!` for the
  exact contraction formulas.
- `nlm = (L_max+1)^2`: total real-SH coefficient count.
- `cos_range`/`sin_range`: contiguous `UnitRange`s into the `nlm` axis holding, for each
  order `m`, the block of "cos"-branch (`m ≥ 0`) or "sin"-branch (`m ≥ 1`) coefficients
  `q_{l,m}`/`q_{l,-m}` for `l = m:L_max` — laid out contiguously so a batched matmul's
  output can be written directly into `qlm_flat[:, cos_range[m+1]]` with no scatter.
"""
struct SHTPlan{FP<:AbstractFloat}
    L_max::Int
    n_theta::Int
    n_phi::Int
    cosθ::Vector{FP}
    gl_weight::Vector{FP}
    φ::Vector{FP}
    u_nodes::Matrix{FP}
    quad_weight::Vector{FP}
    Plm_fwd::Vector{Matrix{FP}}
    Plm_inv::Vector{Matrix{FP}}
    nlm::Int
    cos_range::Vector{UnitRange{Int}}
    sin_range::Vector{UnitRange{Int}}
end

function SHTPlan(L_max::Int, ::Type{FP}=Float64) where FP<:AbstractFloat
    n_theta = L_max + 1
    n_phi = 2 * L_max + 1
    x, w = gauss_legendre_nodes(n_theta, FP)
    φ = FP(2) * FP(pi) .* FP.(0:n_phi-1) ./ n_phi

    Ptab = assoc_legendre_table(x, L_max)

    Plm_fwd = Vector{Matrix{FP}}(undef, L_max + 1)
    Plm_inv = Vector{Matrix{FP}}(undef, L_max + 1)
    for m in 0:L_max
        αm = m == 0 ? one(FP) : sqrt(FP(2))
        n_l = L_max - m + 1
        fwd = zeros(FP, n_theta, n_l)
        inv_ = zeros(FP, n_theta, n_l)
        for (li, l) in enumerate(m:L_max)
            Nlm = sqrt((2l + 1) / (4 * FP(pi)) * _legendre_factorial_ratio(l, m, FP))
            for j in 1:n_theta
                Plmj = Ptab[m+1][j, li]
                inv_[j, li] = αm * Nlm * Plmj
                fwd[j, li] = 2 * FP(pi) * αm * Nlm * w[j] * Plmj
            end
        end
        Plm_fwd[m+1] = fwd
        Plm_inv[m+1] = inv_
    end

    cos_range = Vector{UnitRange{Int}}(undef, L_max + 1)
    sin_range = Vector{UnitRange{Int}}(undef, L_max)
    idx = 1
    cos_range[1] = idx:(idx + L_max)
    idx += L_max + 1
    for m in 1:L_max
        n_l = L_max - m + 1
        cos_range[m+1] = idx:(idx + n_l - 1)
        idx += n_l
        sin_range[m] = idx:(idx + n_l - 1)
        idx += n_l
    end
    nlm = idx - 1
    @assert nlm == (L_max + 1)^2

    n_orient = n_theta * n_phi
    u_nodes = zeros(FP, 3, n_orient)
    quad_weight = zeros(FP, n_orient)
    dφ = 2 * FP(pi) / n_phi
    for jφ in 1:n_phi, jθ in 1:n_theta
        idxu = jθ + (jφ - 1) * n_theta
        cθ = x[jθ]
        sθ = sqrt(max(1 - cθ^2, zero(FP)))
        φj = φ[jφ]
        u_nodes[1, idxu] = sθ * cos(φj)
        u_nodes[2, idxu] = sθ * sin(φj)
        u_nodes[3, idxu] = cθ
        quad_weight[idxu] = w[jθ] * dφ
    end

    return SHTPlan{FP}(L_max, n_theta, n_phi, x, w, φ, u_nodes, quad_weight,
                        Plm_fwd, Plm_inv, nlm, cos_range, sin_range)
end

"""
    sht_forward!(qlm, q_grid, sht::SHTPlan)

Forward real-spherical-harmonic transform. `q_grid` has shape `(spatial_dims...,
n_orient)` (orientation axis last, `n_orient = sht.n_theta*sht.n_phi`); `qlm` has shape
`(spatial_dims..., sht.nlm)`. Batched over `spatial_dims`: one real FFT along `φ`
(reshaping the trailing `n_orient` axis into `(n_theta,n_phi)`), giving raw complex
Fourier coefficients `Ĉ_m(θ_j) = Σ_k q(θ_j,φ_k) e^{-imφ_k}` for `m=0:L_max` (exactly
`L_max+1` frequency bins, by construction of `n_phi`), followed by one dense matmul per
order `m` contracting the `θ` axis against `sht.Plm_fwd[m+1]` — the spherical-harmonic
analogue of a single Fourier-kernel multiply. Writes into `qlm`'s contiguous
`cos_range[m+1]`/`sin_range[m]` coefficient blocks directly (no scatter).
"""
function sht_forward!(qlm::AbstractArray{FP}, q_grid::AbstractArray{FP}, sht::SHTPlan{FP}) where FP<:AbstractFloat
    nd = ndims(q_grid) - 1
    spatial_size = size(q_grid)[1:nd]
    Nsp = prod(spatial_size)
    qg = reshape(q_grid, spatial_size..., sht.n_theta, sht.n_phi)
    C = rfft(qg, nd + 2)
    qlm_flat = reshape(qlm, Nsp, sht.nlm)

    for m in 0:sht.L_max
        Cm = selectdim(C, nd + 2, m + 1)
        am = reshape(real(Cm), Nsp, sht.n_theta) ./ sht.n_phi
        mul!(view(qlm_flat, :, sht.cos_range[m+1]), am, sht.Plm_fwd[m+1])
        if m >= 1
            bm = reshape(.-imag(Cm), Nsp, sht.n_theta) ./ sht.n_phi
            mul!(view(qlm_flat, :, sht.sin_range[m]), bm, sht.Plm_fwd[m+1])
        end
    end
    return qlm
end

"""
    sht_inverse!(q_grid, qlm, sht::SHTPlan)

Inverse real-spherical-harmonic transform (exact inverse of `sht_forward!` for any
`qlm` — the transform pair is exact, not merely approximate, since the `n_theta`-point
Gauss-Legendre quadrature is exact for the degree-`≤2L_max` polynomials involved). One
dense matmul per order `m` (contracting the spherical-harmonic-degree axis against
`sht.Plm_inv[m+1]`) reconstructs each Fourier mode's `θ`-dependence, assembled into the
raw complex Fourier array `Ĉ_m(θ_j) = n_phi·(a_m - i b_m)` (`m=0` has no factor of `1/2`;
`m≥1` does, matching the real/complex-FFT normalization asymmetry between the DC term
and every other harmonic), then one real inverse FFT along `φ` reconstructs `q_grid`.
"""
function sht_inverse!(q_grid::AbstractArray{FP}, qlm::AbstractArray{FP}, sht::SHTPlan{FP}) where FP<:AbstractFloat
    nd = ndims(qlm) - 1
    spatial_size = size(qlm)[1:nd]
    Nsp = prod(spatial_size)
    qlm_flat = reshape(qlm, Nsp, sht.nlm)
    nfreq = sht.L_max + 1

    C = zeros(Complex{FP}, spatial_size..., sht.n_theta, nfreq)
    Cflat = reshape(C, Nsp, sht.n_theta, nfreq)

    Ccos0 = qlm_flat[:, sht.cos_range[1]] * sht.Plm_inv[1]'
    Cflat[:, :, 1] .= complex.(sht.n_phi .* Ccos0)

    for m in 1:sht.L_max
        Ccos = qlm_flat[:, sht.cos_range[m+1]] * sht.Plm_inv[m+1]'
        Csin = qlm_flat[:, sht.sin_range[m]] * sht.Plm_inv[m+1]'
        Cflat[:, :, m+1] .= (sht.n_phi / 2) .* (Ccos .- im .* Csin)
    end

    qg = irfft(C, sht.n_phi, nd + 2)
    q_grid .= reshape(qg, spatial_size..., sht.n_theta * sht.n_phi)
    return q_grid
end

"""
    _orientation_marginalize(qslice, quad_weight, orient_dim)

Contracts one axis of size `length(quad_weight)` at position `orient_dim` against S²
quadrature weights, leaving that axis dropped — `∫du f(r,u) ≈ Σ_i quad_weight[i]
f(r,u_i)`. Used wherever a `WLCPropagator`-produced quantity indexed by orientation must
be marginalized down to a purely positional one (e.g. `q_in(r,u,k)·q_out(r,u,k)` in a
density formula — the elementwise product must happen *before* this contraction, since
`∫du f·g ≠ (∫du f)(∫du g)`).
"""
function _orientation_marginalize(qslice::AbstractArray, quad_weight::AbstractVector, orient_dim::Int)
    w = reshape(quad_weight, ntuple(_ -> 1, orient_dim - 1)..., length(quad_weight))
    return dropdims(sum(qslice .* w; dims=orient_dim); dims=orient_dim)
end

"""
    _Q_PAIRS, _Q_PAIR_WEIGHT

The 6 independent components of a traceless symmetric 3×3 tensor `Q_ab` (`a≤b`), used
wherever a full nematic order-parameter tensor `Q_ab(r) = (3⟨u_a u_b⟩ - δ_ab)/2` (rather
than `orientation_order_parameter`'s single-axis scalar reduction) is needed — e.g. for
a Maier-Saupe mean field. `_Q_PAIR_WEIGHT[j]` is the multiplicity pair `j` carries in a
full double-sum contraction `A:B = Σ_{a,b=1}^3 A_ab B_ab`: off-diagonal pairs occur twice
(once as `(a,b)`, once as `(b,a)`) since both `Q_ab` and any tensor it contracts against
here are symmetric, so `A:B = Σ_j _Q_PAIR_WEIGHT[j] * A[j] * B[j]` over just the 6 stored
components.
"""
const _Q_PAIRS = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
const _Q_PAIR_WEIGHT = (1, 1, 1, 2, 2, 2)

"""
    bending_eigenvalues(κ::Real, L_max::Int) -> Vector{Float64}

Spherical-harmonic-degree eigenvalues `κ̂_l` (`l=0:L_max`, 1-indexed as `κ̂[l+1]`) of the
Kratky-Porod/von-Mises-Fisher bending kernel `K(cosγ) = exp(κ cosγ)/Z(κ)`
(`Z(κ)=4π i_0(κ)`, `i_l` the modified spherical Bessel function of the first kind).
Since `K` depends only on `u_{k-1}·u_k`, its Legendre expansion
`K(cosγ) = Σ_l (2l+1) [i_l(κ)/(4π i_0(κ))] P_l(cosγ)` is diagonal in spherical-harmonic
degree `l`, independent of order `m` — the orientation-space analogue of
`DiscreteGaussianChainPropagator`'s Gaussian kernel being diagonal in Fourier `|k|` —
with eigenvalue `κ̂_l = i_l(κ)/i_0(κ)`. `κ̂_1` is also exactly the bond-orientation
correlation `⟨u_{k-1}·u_k⟩` (the Langevin function `L(κ) = coth(κ) - 1/κ`), used directly
by the discrete freely-rotating-chain end-to-end-distance formula (`test/test_wlc.jl`).
`κ=0` (fully flexible/isotropic bond) gives `κ̂_0=1`, `κ̂_l=0` for `l≥1`. `κ=Inf` (exact
rigid-rod limit, zero reorientation freedom) gives `κ̂_l=1` for every `l`, since
`i_l(κ)/i_0(κ) → 1` as `κ→∞` for any fixed `l` — the bending kernel becomes the identity
operator exactly, with no `L_max`-dependent resolution error.

`i_l(κ) = sqrt(π/(2κ)) besseli(l+1/2, κ)` grows like `e^κ`, overflowing `Float64` for
`κ ≳ 700` — since only the ratio `i_l(κ)/i_0(κ)` is ever needed, and the `sqrt(π/(2κ))`
prefactor and `e^κ` growth are common to every `l`, both cancel exactly, so this uses
`SpecialFunctions.besselix` (the exponentially-scaled Bessel function,
`besselix(ν,κ) = besseli(ν,κ)e^{-κ}`) for both numerator and denominator — overflow-safe
at any finite `κ`, with the `e^κ` factors cancelling before either side is ever formed.
`besselix` itself throws for `κ=Inf` (and for very large finite `κ`, e.g. `1e10`), so
`κ=Inf` is handled as its own early-return rather than falling through to `besselix`.
"""
function bending_eigenvalues(κ::Real, L_max::Int)
    FT = float(typeof(κ))
    κ = FT(κ)
    κ̂ = zeros(FT, L_max + 1)
    κ̂[1] = one(FT)
    κ == zero(FT) && return κ̂
    isinf(κ) && return ones(FT, L_max + 1)
    i0x = besselix(FT(0.5), κ)
    for l in 1:L_max
        ilx = besselix(FT(l) + FT(0.5), κ)
        κ̂[l+1] = ilx / i0x
    end
    return κ̂
end
