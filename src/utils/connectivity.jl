"""
    MolStructure

Abstract type for specifying the group connectivity of a single component, used via the
`mol_structure::Dict{String,<:MolStructure}` keyword accepted by heterosegmented
group-contribution models (e.g. `HeterogcPCPSAFT`). Concrete subtypes are `SMILESStructure`
(built via `smiles`) and `CustomStructure` (built via `custom_structure`).
"""
abstract type MolStructure end

"""
    SMILESStructure <: MolStructure

Wraps a raw SMILES string. Constructed via `smiles(s)`. When resolving connectivity for a
component mapped to a `SMILESStructure`, the group topology is fragmented directly from
the given SMILES string using the `GCIdentifier`/`ChemicalIdentifiers` extension (both
packages must be loaded), rather than looked up by component name.
"""
struct SMILESStructure <: MolStructure
    smiles::String
end

"""
    smiles(s::String)

Construct a `SMILESStructure` from a raw SMILES string `s`, for use as a value in the
`mol_structure` dictionary passed to a heterosegmented group-contribution `DFTSystem`
(e.g. `mol_structure = Dict("1-butanol" => smiles("CCCCO"))`). Requires the
`GCIdentifier`/`ChemicalIdentifiers` extension to be loaded to actually resolve the
connectivity.
"""
smiles(s::String) = SMILESStructure(s)

"""
    CustomStructure <: MolStructure

Wraps a hand-written single-letter-per-group connectivity string (see `custom_structure`).
Unlike `SMILESStructure`, resolving a `CustomStructure`'s connectivity requires no external
chemistry lookup — it is parsed directly.
"""
struct CustomStructure <: MolStructure
    string::String
end

"""
    custom_structure(s::String)

Construct a `CustomStructure` describing a synthetic bead topology directly, without any
chemistry lookup — useful for pseudo-components (e.g. block copolymers) that aren't real,
identifiable molecules. Each non-parenthesis character in `s` is one group instance,
tangentially bonded to the previous one in sequence; parentheses open/close a branch off
the current group. For example, `custom_structure("AAAAABBBB")` describes a linear chain of
5 `A` beads followed by 4 `B` beads, while `custom_structure("A(B)CCC")` describes a `B`
branching off the first `A`, followed by a linear `C-C-C` continuing from that same `A`.
"""
custom_structure(s::String) = CustomStructure(s)

# Parser for single-letter group notation with SMILES-style branching.
# Each non-paren character = one group instance; parentheses open/close branches.
# Example: "AAAAABBBB" -> linear A₁-...-A₅-B₁-...-B₄
#          "A(B)CCC"   -> B branches off A, then C-C-C continues from A
function get_connectivity(::EoSModel, cs::CustomStructure)
    s = cs.string
    group_names = Char[]
    bonds = Pair{Int,Int}[]
    stack = Int[]
    current = 0

    for ch in s
        if ch == '('
            push!(stack, current)
        elseif ch == ')'
            current = pop!(stack)
        else
            push!(group_names, ch)
            idx = length(group_names)
            if current != 0
                push!(bonds, current => idx)
            end
            current = idx
        end
    end

    n = length(group_names)
    bond_mat = zeros(Int64, n, n)
    for (i, j) in bonds
        bond_mat[i, j] = 1
        bond_mat[j, i] = 1
    end

    names = string.(group_names)
    return 1:n, names, bond_mat
end

# Dispatches to the GCIdentifier/ChemicalIdentifiers extension (if loaded) to resolve GC
# connectivity directly from a chemical name. This is *not* implemented as a competing
# `ClassicalDFT.get_connectivity(::EoSModel, ::String)` method inside the extension itself: Julia
# disallows two modules defining an identical method signature during extension
# precompilation, so the extension instead defines a plainly-named `get_connectivity_from_name`
# that this fallback looks up and calls via `Base.get_extension` at runtime.
function get_connectivity(model::EoSModel, name::String)
    ext = Base.get_extension(@__MODULE__, :GCIdentifierClassicalDFTExt)
    if !isnothing(ext)
        return ext.get_connectivity_from_name(model, name)
    end
    error("""
    Auto-detection of GC connectivity from a chemical name requires GCIdentifier and
    ChemicalIdentifiers. Either:
      • load both packages before constructing the DFTSystem, or
      • supply connectivity manually via `mol_structure`:
          DFTSystem(model, structure, options;
              mol_structure = Dict("$(name)" => custom_structure("AAAAABBBB")))
          DFTSystem(model, structure, options;
              mol_structure = Dict("$(name)" => smiles("CCCCCC")))
    """)
end
