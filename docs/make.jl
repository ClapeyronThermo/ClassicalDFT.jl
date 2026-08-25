push!(LOAD_PATH,"../src/")
using Documenter, DocumenterVitepress, Clapeyron, ClassicalDFT

makedocs(sitename = "ClassicalDFT.jl",
format = DocumenterVitepress.MarkdownVitepress(
    repo = "github.com/ClapeyronThermo/ClassicalDFT.jl",
    devbranch = "main",
    devurl = "dev"),
warnonly = Documenter.except(),
# ClassicalDFT re-exports/documents several bindings (e.g. `PCSAFT`) that are really Clapeyron's own
# types. Clapeyron is installed via Pkg.add (a registry tarball, no local .git), so
# Documenter can't auto-detect a remote to build "view source" links from — register it
# manually (branch is a placeholder for URL construction only, not load-bearing).
remotes = Dict(
    dirname(dirname(pathof(Clapeyron))) => (Documenter.Remotes.GitHub("ClapeyronThermo", "Clapeyron.jl"), "master"),
),
    authors = "Pierre J. Walker and Andrés Riedemann.",
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Tutorials" => Any[
        "Getting Started" => "tutorials/getting_started.md",
        "Choosing a Geometry & Adsorption" => "tutorials/geometries.md",
        "Vapour-Liquid Interfaces" => "tutorials/vapor_liquid_interfaces.md",
        "Multi-Dimensional Interfaces" => "tutorials/multidimensional_interfaces.md",
        "Group-Contribution & Heterosegmented Chains" => "tutorials/group_contribution_chains.md",
        "Copolymer Microphase Morphologies" => "tutorials/copolymer_morphology.md",
        "Self-Consistent Field Theory" => "tutorials/scft.md",
        "Electrolytes" => "tutorials/electrolytes.md",
        "Dynamic DFT" => "tutorials/dynamic_dft.md",
        "GPU Acceleration" => "tutorials/gpu_acceleration.md",
        ],
        "Available Models" => Any[
        "SAFT-based Models" => "models/saft.md",
        "Other Functionals" => "models/other.md",
        "Electrolytes" => "models/electrolytes.md",
        "SCFT" => "models/scft.md",
        ],
        "Structures & External Fields" => Any[
        "Structures" => "structures.md",
        "External Fields" => "external_fields.md",
        ],
        "API" => Any[
        "System" => "api/system.md",
        "Methods" => "api/methods.md",
        "Fields" => "api/fields.md",
        "Propagators" => "api/propagators.md",
        "Free Energy Evaluation" => "api/free_energy.md",
        "Utils" => "api/utils.md",
        "Options" => "api/options.md"
        ],
        "FAQ" => "faq.md",
        ])

        DocumenterVitepress.deploydocs(;
    repo = "github.com/ClapeyronThermo/ClassicalDFT.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
