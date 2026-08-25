"""
    DFTStructure{Dim, Coord, Type} <: Any

Abstract supertype for all DFT structures. All concrete structure types, including [`Structure`](@ref), subtype this.

# Type parameters
- `Dim`   : Spatial dimension (1, 2, or 3).
- `Coord` : Coordinate system, subtype of [`DFTCoordType`](@ref).
- `Type`  : System type, subtype of [`DFTTopology`](@ref).

The interface defined by this abstract type is extensible; new coordinate systems or system types can be added by subtyping [`DFTCoordType`](@ref) and [`DFTTopology`](@ref), and by dispatching on `DFTStructure`.
"""
abstract type DFTStructure{Dim,Coord,Type} end

"""
    DFTCoordType

Abstract supertype for coordinate systems. Built‑in implementations are [`Cartesian`](@ref), [`Cylindrical`](@ref), and [`Spherical`](@ref).
"""
abstract type DFTCoordType end

"""
    DFTTopology

Abstract supertype for the specific system topology (e.g. uniform, two‑phase interface, block‑copolymer morphology). Concrete subtypes include [`UniformGrid`](@ref), [`TwoPhaseSystem`](@ref), and [`BlockCopolymerMorphology`](@ref).
"""
abstract type DFTTopology end

"""
    Cartesian <: DFTCoordType

Cartesian (rectilinear) coordinates. Used as the `Coord` type parameter for [`Structure`](@ref).
"""
struct Cartesian <: DFTCoordType end

"""
    Cylindrical <: DFTCoordType

Cylindrical coordinates (radial only, assuming translational invariance along the cylinder axis and rotational symmetry). Used as the `Coord` type parameter for [`Structure`](@ref).
"""
struct Cylindrical <: DFTCoordType end

"""
    Spherical <: DFTCoordType

Spherical coordinates (radial only, assuming full spherical symmetry). Used as the `Coord` type parameter for [`Structure`](@ref).

"""
struct Spherical <: DFTCoordType end

"""
    Structure{Dim, Coord, Top} <: DFTStructure{Dim, Coord, Top}

The central concrete container for all DFT systems. It holds the thermodynamic conditions, spatial bounds, grid definition, bulk densities, and the topology of a specific physical system type.

All concrete system constructors (e.g. [`Uniform1DCart`](@ref), [`TwoPhase1DCart`](@ref), [`LamellarStack3DCart`](@ref)) ultimately build an instance of this struct.

# Type parameters
- `Dim`     : Spatial dimension (1, 2, or 3).
- `Coord`   : Coordinate system; must be a subtype of [`DFTCoordType`](@ref).
              Built‑in options: [`Cartesian`](@ref), [`Cylindrical`](@ref), [`Spherical`](@ref).
- `Top` : The concrete type of the `topology` field; must be a subtype of [`DFTTopology`](@ref).

# Fields
- `conditions::Tuple{Float64,Float64}`
  Thermodynamic conditions: `(pressure, temperature)`.

- `bounds::NTuple{Dim,NTuple{2,Float64}}`
  Lower and upper bounds for each spatial dimension, internally stored as a normalized `NTuple`.
  For radial coordinates (spherical/cylindrical), the second element of the radial bound is used as the aperture of the Hankel transform; the first element is retained (e.g. for excluded‑volume walls) but does not truncate the radial grid.

- `ρbulk::Vector{Float64}`
  Bulk (reference) densities for each species in the system. For multi‑phase structures this typically corresponds to the first (reference) phase.

- `ngrid::NTuple{Dim,Int64}`
  Number of grid points along each dimension. If a number is used instead, the grid points will be equal for all dimensions.

- `topology::Top`
  An object that encodes the particular topology of the system. Examples:
  - [`UniformGrid`](@ref)`()` : a uniform (bulk) system.
  - [`TwoPhaseSystem`](@ref)`{Symbol}(ρbulk2)` : a two‑phase interface.
  - [`BlockCopolymerMorphology`](@ref)`{Symbol}(core_groups, amplitude, periods)` : a periodic microphase morphology.

## Bound types

The bound type is flexible and is normalized at construction time, some examples of supported bounds are:

```julia
# 1D bounds (Dim = 1)
bounds_1d = [-10.0, 10.0]                     # Vector of two numbers
bounds_1d_tuple = (-10.0, 10.0)               # Tuple of two numbers

# 2D bounds (Dim = 2)
bounds_2d_matrix = [0.0 10.0; 0.0 20.0]      # 2×2 matrix (rows = dimensions)
bounds_2d_tuple = ((0.0, 10.0), (0.0, 20.0)) # NTuple{2,NTuple{2,Float64}}

# 3D bounds (Dim = 3)
bounds_3d_matrix = [0.0 10.0; 0.0 20.0; 0.0 30.0]  # 3×2 matrix
bounds_3d_tuple = ((0.0, 10.0), (0.0, 20.0), (0.0, 30.0))
```

!!! note "Constructor"
    While you can call `Structure{Dim,Coord}(conditions,bounds,ρbulk,ngrid,topology)` directly, we recommend the use of the available convenience constructors, such as [`Uniform1DCart`](@ref), [`TwoPhase1DCart`](@ref), [`LamellarStack3DCart`](@ref), etc.
    These functions validate and normalise the inputs, then build the appropriate `Structure` instance. For cylindrical and spherical coordinates, the constructors also build the necessary radial transform structures.

# Examples
```julia
# Uniform 1D Cartesian system
struct = Uniform1DCart((1.0, 300.0), [0.033], [-10.0, 10.0], 201)

# Two‑phase interface in 2D (lamellar)
struct = TwoPhase2DLamCart((0.1, 350.0), [0.03], [0.001],
                           [0.0 20.0; 0.0 10.0], (256, 128))
```
"""
struct Structure{Dim,Coord,Top} <: DFTStructure{Dim,Coord,Top}
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    bounds::NTuple{Dim,NTuple{2,Float64}}
    ngrid::NTuple{Dim,Int64}
    topology::Top
end

function Structure{Dim,Coord}(conditions, ρbulk, bounds, ngrid, topology) where {Dim,Coord}
    assert_ngrid(ngrid,Val(Dim))
    assert_bounds(bounds,Val(Dim))
    norm_bounds = normalize_bounds(bounds, Val(Dim))
    norm_ngrid = normalize_ngrid(ngrid, Val(Dim))
    return Structure{Dim,Coord,typeof(topology)}(conditions, ρbulk, norm_bounds, norm_ngrid, topology)
end

#a way to check the struct first: DFTStructByCoord{Cartesian} = DFTStructure{N,Cartesian}
"""
    DFTStructByCoord{T,N,S} = DFTStructure{N,T,S} where {T,N,S}

Type alias that orders the type parameters by coordinate system first.
Equivalent to `DFTStructure{N,T,S}` but with `Coord` as the first parameter.
Useful for dispatch when coordinate system is the primary discriminator.
"""
const DFTStructByCoord{T,N,S} = DFTStructure{N,T,S} where {T,N,S}

"""
    DFTStructByType{S,N,T} = DFTStructure{N,T,S} where {S,N,T}

Type alias that orders the type parameters by topology first.
Equivalent to `DFTStructure{N,T,S}` but with `S` as the first parameter.
Useful for dispatch when the system type is the primary discriminator.
"""
const DFTStructByType{S,N,T} = DFTStructure{N,T,S} where {S,N,T}

"""
    dimension(x::DFTStructure) -> Int

Return the spatial dimension `Dim` of the structure.
"""
dimension(x::DFTStructure{N}) where N = N
dimension(::Type{<:DFTStructure{N}}) where N = N

"""
    bounds(structure::DFTStructure, dim::Int) -> NTuple{2,Float64}

Return the lower and upper bounds for the given dimension `dim` of the structure.
"""
bounds(structure::DFTStructure,dim::Int = 1) = structure.bounds[dim]

"""
    ngrid(structure::DFTStructure, dim::Int) -> Int

Return the number of grid points for the given dimension `dim` of the structure.
"""
ngrid(structure::DFTStructure,dim = 1) =  structure.ngrid[dim]

"""
    uniform_range(structure::DFTStructure, dim::Int = 1) -> AbstractRange

Return the `N` grid coordinates covering the bounds of the given dimension, using the
number of grid points stored in the structure. Uses the standard **half-open** FFT
convention (`N` samples spanning a period of exactly `ub-lb`, spacing `Δ=(ub-lb)/N`,
matching `structure_fftfreq`/`fftfreq(N, N/(ub-lb))` and `structure_dz`'s own
`(ub-lb)/ngrid`) — *not* `LinRange(lb,ub,N)`, which would put a spacing of
`(ub-lb)/(N-1)` and duplicate the periodic-image point at `ub` (`fftfreq`/FFTW's DFT
definition assumes `N` samples span `N*Δ` exactly, with no redundant endpoint — see
`numpy.fft.fftfreq`'s docs, which use the same underlying DFT convention as FFTW).
"""
function uniform_range(structure::DFTStructure,dim::Int)
    lb,ub = bounds(structure,dim)
    N = ngrid(structure,dim)
    Δ = (ub-lb)/N
    return range(lb, step=Δ, length=N)
end

#= utils =#

assert_ngrid(::Number, ::Val{N}) where N = nothing
assert_ngrid(::NTuple{D}, ::Val{D}) where D = nothing
assert_ngrid(::Tuple, ::Val{D}) where D = throw(DimensionMismatch("Tuple length does not match $D"))

function assert_ngrid(arr::AbstractArray, ::Val{D}) where D
    (length(arr) == D) || throw(DimensionMismatch("Array length does not match $D"))
    return nothing
end

assert_ngrid(::Any, ::Val) = throw(DimensionMismatch("Invalid grid type"))

normalize_ngrid(x::Number, ::Val{N}) where N = ntuple(i -> x,Val(N))
normalize_ngrid(t::NTuple{D}, ::Val{D}) where D = t
normalize_ngrid(::Tuple, ::Val{D}) where D = throw(DimensionMismatch("Tuple length does not match $D"))

function normalize_ngrid(arr::AbstractArray, ::Val{D}) where D
    length(arr) == D || throw(DimensionMismatch("Array length does not match $D"))
    return Tuple(arr)
end
normalize_ngrid(::Any, ::Val) = throw(DimensionMismatch("Invalid grid type"))

function assert_bounds(t::NTuple, ::Val{N}) where N
    if N == 1 && length(t) == 2
        return nothing
    end
    length(t) == N || throw(DimensionMismatch("Tuple length $(length(t)) != $N"))
    for (i, x) in enumerate(t)
        if !(x isa Tuple && length(x) == 2 && x[1] isa Number && x[2] isa Number)
            throw(DimensionMismatch("Element $i is not a 2‑tuple of numbers"))
        end
    end
    return nothing
end

assert_bounds(v::AbstractVector, ::Val{1}) =
    (length(v) == 2 && all(isa(x, Number) for x in v)) ||
    throw(DimensionMismatch("Vector must have exactly 2 numbers for 1‑D bound"))

assert_bounds(::AbstractVector, ::Val{N}) where N =
    throw(DimensionMismatch("Vector is only valid for 1‑D bound, got $N"))

function assert_bounds(m::AbstractMatrix, ::Val{N}) where N
    size(m) == (N, 2) || throw(DimensionMismatch("Matrix size $(size(m)) != ($N, 2)"))
    all(isa(x, Number) for x in m) ||
        throw(DimensionMismatch("Matrix contains non‑numeric values"))
    return nothing
end

assert_bounds(::Any, ::Val) =  throw(DimensionMismatch("Invalid bounds type"))

normalize_bounds(t::NTuple{2}, ::Val{1}) = (t,)
normalize_bounds(t::NTuple{N,NTuple{2,Float64}}, ::Val{N}) where N = t
normalize_bounds(v::AbstractVector, ::Val{1}) = ((v[1], v[2]),)
normalize_bounds(m::AbstractMatrix, ::Val{N}) where N = ntuple(i -> (m[i,1], m[i,2]), Val(N))
normalize_bounds(::Any, ::Val) = throw(DimensionMismatch("Invalid bounds type"))

function _1d_to_2d(bounds, ngrid)
    assert_bounds(bounds, Val(1))
    assert_ngrid(ngrid, Val(1))
    lb, ub = bounds
    n = only(ngrid)
    new_bounds = ((lb, ub), (lb, ub))
    new_ngrid = (n, n)
    return new_bounds, new_ngrid
end

function _1d_to_3d(bounds, ngrid)
    assert_bounds(bounds, Val(1))
    assert_ngrid(ngrid, Val(1))
    lb, ub = bounds
    n = only(ngrid)
    new_bounds = ((lb, ub), (lb, ub), (lb, ub))
    new_ngrid = (n, n, n)
    return new_bounds, new_ngrid
end

function Base.show(io::IO,::MIME"text/plain",structure::DFTStructure{N,C,P}) where {N,C,P}
    print(io,N,"D ")
    p,T = structure.conditions
    compact_io = IOContext(io, :compact => true)
    println(compact_io,typeof(structure),"(p = $p, T = $T):")
    println(compact_io," number of species: ",length(structure.ρbulk))
    if N == 1
        println(compact_io," bounds           : ",only(structure.bounds))
        println(" grid points      : ",only(structure.ngrid))
    else
        println(compact_io," bounds           : ",structure.bounds)
        println(compact_io," grid points      : ",structure.ngrid)
    end
    print(compact_io," topology         : ",structure.topology)
end

"""
    UniformGrid{T} <: DFTTopology

System topology for uniform (bulk) systems.
The `transform` field may hold precomputed radial transform data (e.g. for spherical/cylindrical coordinates); for Cartesian systems it is typically `nothing`.
Constructors that produce uniform systems, such as [`Uniform1DCart`](@ref), [`Uniform1DSphr`](@ref), etc., return a [`Structure`](@ref) with `topology = UniformGrid(...)`.
"""
struct UniformGrid{T} <: DFTTopology
    transform::T
end

UniformGrid() = UniformGrid(nothing)

Base.show(io::IO,x::UniformGrid) = print(io,typeof(x),"()")

"""
    Uniform1DCart(conditions, ρbulk, bounds, ngrid)

Construct a uniform (bulk) system in 1D Cartesian coordinates. This is the simplest system type and is primarily used to benchmark DFT implementations against bulk (uniform) thermodynamic predictions.

Returns a [`Structure`](@ref) with `Dim = 1`, `Coord = Cartesian`, and `topology = UniformGrid()`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk density of each species.
- `bounds`                             : 1D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid`                              : Number of grid points.

# Example
```julia
julia> structure = Uniform1DCart((1.0, 300.0), [0.033], [-10.0, 10.0], 201)
```
"""
function Uniform1DCart(conditions,ρbulk,bounds,ngrid)
    return Structure{1,Cartesian}(conditions,ρbulk,bounds,ngrid,UniformGrid())
end

"""
    Uniform2DCart(conditions, ρbulk, bounds, ngrid)

Construct a uniform (bulk) system in 2D Cartesian coordinates.

Returns a [`Structure`](@ref) with `Dim = 2`, `Coord = Cartesian`, and `topology = UniformGrid()`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk density of each species.
- `bounds`                             : 2D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid`                              : Number of grid points for each dimension. If a number is passed, then all dimensions will have the same number of grid points.

# Example
```julia
julia> structure = Uniform2DCart((1.0, 300.0), [0.033], [-10.0 10.0; -10.0 10.0], 101)
```
"""
Uniform2DCart(conditions, ρbulk, bounds, ngrid) = Structure{2,Cartesian}(conditions,ρbulk,bounds,ngrid,UniformGrid())

"""
    Uniform3DCart(conditions, ρbulk, bounds, ngrid)

Construct a uniform (bulk) system in 3D Cartesian coordinates.

Returns a [`Structure`](@ref) with `Dim = 3`, `Coord = Cartesian`, and `topology = UniformGrid()`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk density of each species.
- `bounds`                             : 3D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid`                              : Number of grid points for each dimension. If a number is passed, then all dimensions will have the same number of grid points.

# Example
```julia
julia> structure = Uniform3DCart((1.0, 300.0), [0.033], [-10.0 10.0; -10.0 10.0; -10.0 10.0], 51)
```
"""
Uniform3DCart(conditions, ρbulk, bounds, ngrid) = Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,UniformGrid())

#=================

Radial transforms over uniform grids

=================#

radial_transform(structure::DFTStructure{<:Any,<:Any,<:UniformGrid}) = radial_transform(structure.topology)
radial_transform(data::UniformGrid{Q}) where Q = data.transform

#generators: from uniform grid to the adequate radial transform

function to_radial(s0::DFTStructure{1,Spherical,UniformGrid{Nothing}})
    _,ub = bounds(s0)
    grid = UniformGrid(Hankel.QDHT(0, 2, ub, ngrid(s0)))
    Structure{1,Spherical,typeof(grid)}(s0.conditions,s0.ρbulk,s0.bounds,s0.ngrid,grid)
end

function to_radial(s0::DFTStructure{1,Cylindrical,UniformGrid{Nothing}})
    _,ub = bounds(s0)
    grid = UniformGrid(Hankel.QDHT(0, 1, ub, ngrid(s0)))
    Structure{1,Cylindrical}(s0.conditions,s0.ρbulk,s0.bounds,s0.ngrid,grid)
end

"""
    RadialFrequency{FP,Q}

Wraps the `Hankel.QDHT` used by a `Uniform1DSphr`/`Uniform1DCyl` structure together with the corresponding "ordinary frequency" vector `ω̄ = Q.k ./ 2π`.
Using this convention (rather than `Q.k` directly) means the *same* closed-form kernel formulas already used for the Cartesian FFT path (e.g. `sin(ω̄R)/ω̄`-style expressions, with `R = 2π*width`) can be reused unchanged for the radial case, just substituting this `ω̄` in place of `sqrt.(sum(abs2,ω,dims=nd+1))`.
Returned by `structure_ω` for spherical/cylindrical structures in place of the raw `Array{Complex}` returned for Cartesian ones; this lets `SWeightedDensity`/`VWeightedDensity`/etc. dispatch to the correct (QDHT-based) constructor without any change to the models' `get_fields` call sites, which just pass `ω` through positionally.
"""
struct RadialFrequency{FP<:AbstractFloat, Q<:Hankel.QDHT}
    Q::Q
    ω̄::Vector{FP}
end

"""
    Uniform1DSphr(conditions, ρbulk, bounds, ngrid)

Construct a uniform (bulk) system in 1D spherical coordinates (radial only, full spherical symmetry). The radial grid spans from near `0` to `ub`, where `ub` is taken from the second element of `bounds`.
The first element `lb` (if given) is stored for external field placement but does not truncate the grid.

Returns a [`Structure`](@ref) with `Dim = 1`, `Coord = Spherical`, and `topology = UniformGrid(Hankel.QDHT(...))` (a precomputed Hankel transform object).

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk density of each species.
- `bounds`                             : `[lb, ub]` or just `ub`
- `ngrid`                              : Number of radial grid points.

# Example
```julia
julia> structure = Uniform1DSphr((1.0, 300.0), [0.033], [0.0, 10.0], 201)
julia> structure = Uniform1DSphr((1.0, 300.0), [0.033], 10.0, 201)
```
"""
Uniform1DSphr(conditions,ρbulk,bounds,ngrid) = to_radial(Structure{1,Spherical}(conditions,ρbulk,bounds,ngrid,UniformGrid(nothing)))
Uniform1DSphr(conditions,ρbulk,ub::Real,ngrid) = Uniform1DSphr(conditions,ρbulk,(zero(ub),ub),ngrid)

"""
    Uniform1DCyl(conditions, ρbulk, bounds, ngrid)

Construct a uniform (bulk) system in 1D cylindrical coordinates (radial only, translational invariance along the cylinder axis and rotational symmetry).
The radial grid spans from near `0` to `ub`, where `ub` is taken from the  second element of `bounds`. The first element `lb` (if given) is stored for external field placement but does not truncate the grid.

Returns a [`Structure`](@ref) with `Dim = 1`, `Coord = Cylindrical`, and `topology = UniformGrid(Hankel.QDHT(...))`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk density of each species.
- `bounds`                             : `[lb, ub]` or just `ub`
- `ngrid`                              : Number of radial grid points.

# Example
```julia
julia> structure = Uniform1DCyl((1.0, 300.0), [0.033], [0.0, 20.0], 201)
julia> structure = Uniform1DCyl((1.0, 300.0), [0.033], 20.0, 201)
```
"""
Uniform1DCyl(conditions, ρbulk, bounds, ngrid) = to_radial(Structure{1,Cylindrical}(conditions,ρbulk,bounds,ngrid,UniformGrid(nothing)))
Uniform1DCyl(conditions, ρbulk, ub::Real, ngrid) = Uniform1DCyl(conditions,ρbulk,(zero(ub),ub),ngrid)

#=================

Two Phase Systems

The free parameter is a symbol, specifying the type of system of the second phase.
ej: TwoPhaseSystem{:Cartesian}

available symbols are:

- :Cartesian
- :Lamellar
- :HexLattice
- :Spherical
=================#

"""
    TwoPhaseSystem{T} <: DFTTopology

Topology type for two‑phase interfaces. The type parameter `T` is a `Symbol` indicating the geometry of the interface:
- `:Cartesian`  : planar interface (1D).
- `:Lamellar`   : lamellar interface in 2D/3D
- `:HexLattice` : hexagonal (cylindrical) domains in 2D/3D.
- `:Spherical`  : spherical droplet/bubble in 3D.

The field `ρbulk2::Vector{Float64}` stores the bulk densities of the second phase.

Constructors such as [`TwoPhase1DCart`](@ref), [`TwoPhase2DLamCart`](@ref), etc. return a [`Structure`](@ref) with this `topology`.
"""
struct TwoPhaseSystem{T} <: DFTTopology
    ρbulk2::Vector{Float64}
end

Base.show(io::IO,top::TwoPhaseSystem) = print(io,typeof(top),"()")
"""
    TwoPhase1DCart(conditions, ρbulk, ρbulk2, bounds, ngrid)

Construct a 1D Cartesian two‑phase interface (e.g. vapour‑liquid or liquid‑liquid).
The interface is positioned at the centre of the box, and the initial density profile is seeded as a sigmoidal transition between `ρbulk` (phase 1) and `ρbulk2` (phase 2).
Returns a [`Structure`](@ref) with `Dim = 1`, `Coord = Cartesian`, and `topology = TwoPhaseSystem{:Cartesian}(ρbulk2)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk densities of the first phase.
- `ρbulk2::Vector{Float64}`            : Bulk densities of the second phase.
- `bounds`                             : 1D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid::Int64`                       : Number of grid points.

!!! note "Fixed boundary densities"
    The densities at the box boundaries are **fixed** to the supplied bulk values (`ρbulk` at the left, `ρbulk2` at the right).
    This ensures that the profiles reach their asymptotic bulk values, which is essential for surface tension calculations.

# Example
```julia
julia> struct = TwoPhase1DCart((p_sat, T), ρ_liquid, ρ_vapour, [-10.0, 10.0], 201)
```
"""
TwoPhase1DCart(conditions,ρbulk,ρbulk2,bounds,ngrid) = Structure{1,Cartesian}(conditions,ρbulk,bounds,ngrid,TwoPhaseSystem{:Cartesian}(ρbulk2))

"""
    TwoPhase2DLamCart(conditions, ρbulk, ρbulk2, bounds, ngrid)

Construct a lamellar two‑phase interface in 2D Cartesian coordinates.
The interface is planar and located along the first dimension, while the second dimension is homogeneous/periodic.

Returns a [`Structure`](@ref) with `Dim = 2`, `Coord = Cartesian`, and `topology = TwoPhaseSystem{:Lamellar}(ρbulk2)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk densities of the first phase.
- `ρbulk2::Vector{Float64}`            : Bulk densities of the second phase.
- `bounds`                             : 2D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid`                              : Number of grid points for each dimension. If a number is passed, then all dimensions will have the same number of grid points.

# Example
```julia
julia> struct = TwoPhase2DLamCart((p, T), ρ1, ρ2, [0.0 20.0; 0.0 10.0], (256, 128))
```
"""
TwoPhase2DLamCart(conditions,ρbulk,ρbulk2,bounds,ngrid) = Structure{2,Cartesian}(conditions,ρbulk,bounds,ngrid,TwoPhaseSystem{:Lamellar}(ρbulk2))

"""
    TwoPhase3DLamCart(conditions, ρbulk, ρbulk2, bounds, ngrid)

Construct a lamellar two‑phase interface in 3D Cartesian coordinates.
The interface is planar along the first dimension; the other two dimensions
are homogeneous/periodic.

Returns a [`Structure`](@ref) with `Dim = 3`, `Coord = Cartesian`, and
`topology = TwoPhaseSystem{:Lamellar}(ρbulk2)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk densities of the first phase.
- `ρbulk2::Vector{Float64}`            : Bulk densities of the second phase.
- `bounds`                             : 3D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid`                              : Number of grid points for each dimension. If a number is passed, then all dimensions will have the same number of grid points.

# Example
```julia
julia> struct = TwoPhase3DLamCart((p, T), ρ1, ρ2, [0.0 20.0; 0.0 10.0; 0.0 10.0], (64, 32, 32))
```
"""
TwoPhase3DLamCart(conditions,ρbulk,ρbulk2,bounds,ngrid) = Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,TwoPhaseSystem{:Lamellar}(ρbulk2))

"""
    TwoPhase2DHexCart(conditions, ρbulk, ρbulk2, bounds, ngrid)

Construct a two‑phase interface with hexagonal (cylindrical‑domain) symmetry in a 2D Cartesian cross‑section.
The profile is seeded as a radially‑symmetric sigmoidal centred on the box, representing a cylindrical domain of phase 1 surrounded by phase 2.
Returns a [`Structure`](@ref) with `Dim = 2`, `Coord = Cartesian`, and `topology = TwoPhaseSystem{:HexLattice}(ρbulk2)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk densities of the first (core) phase.
- `ρbulk2::Vector{Float64}`            : Bulk densities of the second (matrix) phase.
- `bounds`                             : 1D bounds `[lb, ub]` for the square box; the same bounds are used for both dimensions.
- `ngrid::Int`                         : Number of grid points for the square box (equal for all dimensions)

!!! note "Bounds expansion"
    While the generated `Structure` is 2-dimensional (square), This function only takes 1D-input for `bounds` and `ngrid`.
"""
function TwoPhase2DHexCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    bounds_2d, ngrid_2d = _1d_to_2d(bounds, ngrid)
    return Structure{2,Cartesian}(conditions,ρbulk,bounds_2d,ngrid_2d,TwoPhaseSystem{:HexLattice}(ρbulk2))
end

"""
    TwoPhase3DHexCart(conditions, ρbulk, ρbulk2, bounds, ngrid)

Construct a two‑phase interface with hexagonal (cylindrical‑domain) symmetry in 3D Cartesian coordinates. The profile is seeded as a radially‑symmetric sigmoidal in the cross‑section spanned by the first two dimensions, uniform along the third.

Returns a [`Structure`](@ref) with `Dim = 3`, `Coord = Cartesian`, and `topology = TwoPhaseSystem{:HexLattice}(ρbulk2)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`            : Bulk densities of the first (core) phase.
- `ρbulk2::Vector{Float64}`           : Bulk densities of the second (matrix) phase.
- `bounds`                             : 1D bounds `[lb, ub]`for the box; the same bounds are used for all three dimensions.
- `ngrid::Int`                         : Number of grid points for the cube box (equal for all dimensions)

!!! note "Bounds expansion"
    While the generated `Structure` is 3-dimensional (cube), This function only takes 1D-input for `bounds` and `ngrid`.

"""
function TwoPhase3DHexCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    bounds_3d, ngrid_3d = _1d_to_3d(bounds, ngrid)
    return Structure{3,Cartesian}(conditions,ρbulk,bounds_3d,ngrid_3d,TwoPhaseSystem{:HexLattice}(ρbulk2))
end

"""
    TwoPhase3DSphrCart(conditions, ρbulk, ρbulk2, bounds, ngrid)

Construct a two‑phase interface with spherical symmetry (droplet/bubble) embedded in a 3D Cartesian box. The profile is seeded as a radially‑symmetric sigmoidal centred on the box.

Returns a [`Structure`](@ref) with `Dim = 3`, `Coord = Cartesian`, and `topology = TwoPhaseSystem{:Spherical}(ρbulk2)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Bulk densities of the first (interior) phase.
- `ρbulk2::Vector{Float64}`            : Bulk densities of the second (surrounding) phase.
- `bounds`                             : 1D bounds `[lb, ub]` for the cubic box; the same bounds are used for all three dimensions.
- `ngrid::Int`                         : Number of grid points for the cubic box (equal for all dimensions)

!!! note "Bounds expansion"
    While the generated `Structure` is 3-dimensional (cube), This function only takes 1D-input for `bounds` and `ngrid`.
"""
function TwoPhase3DSphrCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    bounds_3d, ngrid_3d = _1d_to_3d(bounds, ngrid)
    return Structure{3,Cartesian}(conditions,ρbulk,bounds_3d,ngrid_3d,TwoPhaseSystem{:Spherical}(ρbulk2))
end

# ── Block-copolymer microphase morphologies ─────────────────────────────────
#
# Unlike the TwoPhaseSystem structure above (one scalar profile per *component*,
# transitioning between two bulk phases), these represent a single periodic unit cell of
# a microphase-separated block-copolymer melt, in which *different groups within the same
# component* (e.g. "A" vs "B" from a `custom_structure`) enrich in different spatial
# domains. `core_groups` names which groups form the minority/core domain (spheres for
# BCC, cylinders for Hex, the network for Gyroid, one set of layers for Lamellar); every
# other group in the model is treated as the surrounding matrix. Requires a
# group-contribution model (anything with `model.groups`, e.g. `HeterogcPCPSAFT`) — see
# `src/structure/morphology.jl` for the seed-profile math.

"""
    BlockCopolymerMorphology{T} <: DFTTopology

System type for block‑copolymer microphase morphologies. The type parameter `T` is a `Symbol` indicating the morphology:
- `:Lamellar`          : alternating layers.
- `:HexLattice`        : hexagonally packed cylinders.
- `:BodyCenteredCubic` : BCC spheres.
- `:Gyroid`            : Schoen gyroid (Ia‑3d).

Fields:
- `core_groups::Vector{String}` : names of the groups that form the minority (core) domains.
- `amplitude::Float64`          : initial seeding amplitude.
- `periods::Int`                : number of unit cells tiled in each dimension.

Constructors such as [`LamellarStack1DCart`](@ref), [`HexLattice2DCart`](@ref), [`BCC3DCart`](@ref), etc. return a [`Structure`](@ref) with this `topology`.
"""
struct BlockCopolymerMorphology{T} <: DFTTopology
    core_groups::Vector{String}
    amplitude::Float64
    periods::Int

    function BlockCopolymerMorphology{T}(core_groups::Vector{String},amplitude::Float64,periods::Int) where T
        periods >= 1 || throw(ArgumentError("periods must be a positive integer, got $periods"))
        return new{T}(core_groups,amplitude,periods)
    end
end

Base.show(io::IO,top::BlockCopolymerMorphology) = Clapeyron.show_as_namedtuple(io,top)

"""
    LamellarStack1DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)

Seed a periodic lamellar (alternating‑layer) block‑copolymer morphology in 1D Cartesian coordinates.
core_groups` enrich in one set of layers; all other groups form the alternating layers.

Returns a [`Structure`](@ref) with `Dim = 1`, `Coord = Cartesian`, and `topology = BlockCopolymerMorphology{:Lamellar}(...)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Average densities for each species.
- `bounds`                             : 1D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid`                              : Number of grid points.
- `core_groups::Vector{String}`        : Group names for the core layers.
- `amplitude::Float64`                 : Seeding amplitude (default `0.3`).
- `periods::Int`                       : Number of lamellar periods seeded initially across the box (default `1`).

# Example
```julia
julia> struct = LamellarStack1DCart((p, T), [0.1], [-10.0, 10.0], 201;
                                    core_groups=["A"], amplitude=0.4, periods=2)
```
"""
function LamellarStack1DCart(conditions,ρbulk,bounds,ngrid; core_groups, amplitude=0.3, periods=1)
    lam = BlockCopolymerMorphology{:Lamellar}(core_groups,amplitude,periods)
    Structure{1,Cartesian}(conditions,ρbulk,bounds,ngrid,lam)
end

"""
    LamellarStack2DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)

2D Cartesian counterpart of [`LamellarStack1DCart`](@ref). Layers alternate along the first dimension and are uniform along the second.

Returns a [`Structure`](@ref) with `Dim = 2`, `Coord = Cartesian`, and `topology = BlockCopolymerMorphology{:Lamellar}(...)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Average densities for each species.
- `bounds`                             : 2D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid::Tuple{Int,Int}`              : Grid points per dimension.
- `core_groups::Vector{String}`        : Group names for the core layers.
- `amplitude::Float64`                 : Seeding amplitude (default `0.3`).
- `periods::Int`                       : Number of lamellar periods seeded initially along the first dimension (default `1`).
"""
function LamellarStack2DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)
    lam = BlockCopolymerMorphology{:Lamellar}(core_groups,amplitude,periods)
    Structure{2,Cartesian}(conditions,ρbulk,bounds,ngrid,lam)
end

"""
    LamellarStack3DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)

3D Cartesian counterpart of [`LamellarStack1DCart`](@ref). Layers alternate along the first dimension and are uniform along the other two.

Returns a [`Structure`](@ref) with `Dim = 3`, `Coord = Cartesian`, and `topology = BlockCopolymerMorphology{:Lamellar}(...)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`            : Average densities for each species.
- `bounds`                             : 3D spatial bounds (see the [`Structure`](@ref) for supported bound inputs).
- `ngrid::Tuple{Int,Int,Int}`         : Grid points per dimension.
- `core_groups::Vector{String}`       : Group names for the core layers.
- `amplitude::Float64`                : Seeding amplitude (default `0.3`).
- `periods::Int`                      : Number of lamellar periods seeded initially along the first dimension (default `1`).
"""
function LamellarStack3DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)
    lam = BlockCopolymerMorphology{:Lamellar}(core_groups,amplitude,periods)
    Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,lam)
end

#=

HexLattice

=#

function assert_hex_bounds(bounds,::Val{N}) where N
    assert_bounds(bounds,Val(N))
    std_bounds = normalize_bounds(bounds,Val(N))
    L = ntuple(i -> std_bounds[i][2] - std_bounds[i][1], Val(N))
    Lx, Ly = L[1], L[2]
    isapprox(Ly, sqrt(3)*Lx; rtol=0.05) || @warn "HexLattice$(N)DCart: Ly = (uby - lby) should span ≈ √3 × Lx = (ubx - lbx) for a clean hexagonal supercell (got Ly/Lx = $(Ly/Lx)); the initial guess will be distorted."
end

"""
    HexLattice2DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)

Seed a periodic hexagonally‑packed cylinder morphology in 2D Cartesian cross‑section.
`core_groups` enrich in the cylindrical domains; other groups form the matrix.
Returns a [`Structure`](@ref) with `Dim = 2`, `Coord = Cartesian`, and `topology = BlockCopolymerMorphology{:HexLattice}(...)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Average densities for each species.
- `bounds`                             : 2D bounds. A warning is issued if the second dimension is not approximately `√3` times the first one (this only affects the quality of the initial guess).
- `ngrid::Tuple{Int,Int}`              : Grid points per dimension.
- `core_groups::Vector{String}`        : Group names for the cylinder domains.
- `amplitude::Float64`                 : Seeding amplitude (default `0.3`).
- `periods::Int`                       : Number of unit cells tiled along each dimension (default `1`).

!!! note "Supercell aspect ratio"
    For a clean hexagonal tiling under periodic (FFT) boundary conditions, the box dimensions (Lx, Ly) should have a ratio near `√3`. A warning is emitted if this ratio deviates by more than 5%.
"""
function HexLattice2DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)
    hex = BlockCopolymerMorphology{:HexLattice}(core_groups,amplitude,periods)
    assert_hex_bounds(bounds,Val(2))
    Structure{2,Cartesian}(conditions, ρbulk, bounds, ngrid, hex)
end

"""
    HexLattice3DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)

3D Cartesian counterpart of [`HexLattice2DCart`](@ref). The hexagonal cylinder lattice is extruded uniformly along the third dimension.

Returns a [`Structure`](@ref) with `Dim = 3`, `Coord = Cartesian`, and `topology = BlockCopolymerMorphology{:HexLattice}(...)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Average densities for each species.
- `bounds`                             : 3D bounds. The first two dimensions should satisfy the hexagonal aspect ratio (see [`HexLattice2DCart`](@ref)); the third is arbitrary.
- `ngrid::Tuple{Int,Int,Int}`          : Grid points per dimension.
- `core_groups::Vector{String}`        : Group names for the cylinder domains.
- `amplitude::Float64`                 : Seeding amplitude (default `0.3`).
- `periods::Int`                       : Number of unit cells tiled along each dimension (default `1`).
"""
function HexLattice3DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)
    hex = BlockCopolymerMorphology{:HexLattice}(core_groups,amplitude,periods)
    assert_hex_bounds(bounds,Val(3))
    Structure{3,Cartesian}(conditions, ρbulk, bounds, ngrid, hex)
end

function assert_cubic_bounds(bounds)
    assert_bounds(bounds,Val(3))
    std_bounds = normalize_bounds(bounds,Val(3))
    L = ntuple(i -> std_bounds[i][2] - std_bounds[i][1], Val(3))
    isapprox(L[1],L[2];rtol=0.02) && isapprox(L[2],L[3];rtol=0.02) || @warn "Cubic Bounds Check: bounds should be cubic (equal extent in all 3 dimensions) for a clean unit cell; got extents $(L)."
end

"""
    BCC3DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)

Seed a periodic body‑centered‑cubic (BCC, Im‑3m) block‑copolymer morphology in 3D Cartesian coordinates.
`core_groups` enrich into spheres at the corners and body‑centre of each cubic unit cell; other groups form the matrix.

Returns a [`Structure`](@ref) with `Dim = 3`, `Coord = Cartesian`, and `topology = BlockCopolymerMorphology{:BodyCenteredCubic}(...)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Average densities for each species.
- `bounds`                             : 3D bounds. A warning is issued if the box is not cubic (equal extents in all dimensions).
- `ngrid::Tuple{Int,Int,Int}`          : Grid points per dimension.
- `core_groups::Vector{String}`        : Group names for the BCC domains.
- `amplitude::Float64`                 : Seeding amplitude (default `0.3`).
- `periods::Int`                       : Number of unit cells tiled along each dimension (default `1`).

# Example
```julia
julia> L = 20.0
julia> struct = BCC3DCart((1.0, 300.0), [0.1], [0.0 L; 0.0 L; 0.0 L], (64, 64, 64);
                          core_groups=["A"], amplitude=0.4, periods=2)
```
"""
function BCC3DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)
    bcc = BlockCopolymerMorphology{:BodyCenteredCubic}(core_groups,amplitude,periods)
    assert_cubic_bounds(bounds)
    Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,bcc)
end

"""
    Gyroid3DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)

Seed a periodic gyroid (Ia‑3d) block‑copolymer morphology using the standard Schoen gyroid level‑set as the initial guess.
`core_groups` enrich on one side of the level set (one of the two interpenetrating networks); other groups enrich on the other.

Returns a [`Structure`](@ref) with `Dim = 3`, `Coord = Cartesian`, and `topology = BlockCopolymerMorphology{:Gyroid}(...)`.

# Arguments
- `conditions::Tuple{Float64,Float64}` : `(pressure, temperature)`.
- `ρbulk::Vector{Float64}`             : Average densities for each species.
- `bounds`                             : 3D bounds. A warning is issued if the box is not cubic.
- `ngrid::Tuple{Int,Int,Int}`          : Grid points per dimension.
- `core_groups::Vector{String}`        : Group names for the core network.
- `amplitude::Float64`                 : Seeding amplitude (default `0.3`).
- `periods::Int`                       : Number of unit cells tiled along each dimension (default `1`).

# Example
```julia
julia> struct = Gyroid3DCart((p, T), [0.1], [0.0 L; 0.0 L; 0.0 L], (64, 64, 64);
                             core_groups=["A"], periods=2)
```
"""
function Gyroid3DCart(conditions, ρbulk, bounds, ngrid; core_groups, amplitude=0.3, periods=1)
    gyr = BlockCopolymerMorphology{:Gyroid}(core_groups,amplitude,periods)
    assert_cubic_bounds(bounds)
    Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,gyr)
end

export Structure, Cartesian, Cylindrical, Spherical
export UniformGrid, TwoPhaseSystem, BlockCopolymerMorphology
export Uniform1DCart, Uniform2DCart, Uniform3DCart
export Uniform1DSphr, Uniform1DCyl
export TwoPhase1DCart, TwoPhase2DLamCart, TwoPhase3DLamCart
export TwoPhase2DHexCart
export LamellarStack1DCart, LamellarStack2DCart, LamellarStack3DCart
export HexLattice2DCart, HexLattice3DCart
export BCC3DCart, Gyroid3DCart