module GCWithSearch

using ClassicalDFT, GCIdentifier, ChemicalIdentifiers
import GCIdentifier: get_expanded_groups, get_mol, get_atoms, __getbondlist, get_grouplist
import Clapeyron: EoSModel

const GCExt = Base.get_extension(ClassicalDFT,:GCIdentifierCDFTExt) 

function ClassicalDFT.get_connectivity_from_name(model::EoSModel, name::String)
    GCExt._get_connectivity_from_smiles(model, search_chemical(name).smiles)
end

end
