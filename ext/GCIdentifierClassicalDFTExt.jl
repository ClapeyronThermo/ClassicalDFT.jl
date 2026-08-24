module GCIdentifierClassicalDFTExt

using ClassicalDFT, GCIdentifier, ChemicalIdentifiers
import GCIdentifier: get_expanded_groups, get_mol, get_atoms, __getbondlist, get_grouplist
import Clapeyron: EoSModel

# Explicit SMILES path — only GCIdentifier needed, not ChemicalIdentifiers
function ClassicalDFT.get_connectivity(model::EoSModel, ss::ClassicalDFT.SMILESStructure)
    _get_connectivity_from_smiles(model, ss.smiles)
end

# Auto-detect from chemical name — both GCIdentifier and ChemicalIdentifiers needed.
# Deliberately *not* named `ClassicalDFT.get_connectivity(::EoSModel, ::String)`: ClassicalDFT's own
# connectivity.jl already defines that exact signature as an error fallback for when this
# extension isn't loaded, and it looks up and calls this function via `Base.get_extension`
# at runtime instead.
function get_connectivity_from_name(model::EoSModel, name::String)
    _get_connectivity_from_smiles(model, search_chemical(name).smiles)
end

function _get_connectivity_from_smiles(model, smiles_str)
    mol = get_mol(smiles_str)
    atoms = get_atoms(mol)
    bondlist = __getbondlist(mol)
    groups = get_grouplist(model)
    group_id, mapping = get_expanded_groups(mol, groups, atoms, bondlist, false, smiles_str)

    bond_mat = zeros(Int64, length(group_id), length(group_id))
    for bond in bondlist
        g1 = mapping[:, bond[1]] .== 1
        g2 = mapping[:, bond[2]] .== 1
        g1 == g2 && continue
        bond_mat[g1, g2] .+= 1
        bond_mat[g2, g1] .+= 1
    end

    group_names = [GCIdentifier.name(groups[i]) for i in group_id]
    return group_id, group_names, bond_mat
end

end
