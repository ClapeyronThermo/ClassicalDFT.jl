"""
    texlabel(s::AbstractString; upright::Bool=false)

Convert a plot label string (as used by `PlotsClassicalDFTExt`/`MakieClassicalDFTExt`) into a LaTeX-rendered
`LaTeXString`. Only implemented once `LaTeXStrings.jl` is loaded — see
`ext/LaTeXStringsClassicalDFTExt.jl`. Called when `latex=true` is passed to `plot(system, ...)`.
`upright=true` wraps the result in `\\mathrm{}` so it renders in upright (non-italic) font —
used for legend entries (species/group names), which are labels rather than math variables.
When `upright=false` (axis labels), a `"quantity / unit"`-style string has its unit half
(after `" / "`) set upright automatically, matching scientific-typesetting convention that
units aren't italicized; the quantity symbol before it stays in the default italic math font.
"""
texlabel(s::AbstractString; upright::Bool=false) = error("latex=true requires LaTeXStrings.jl to be loaded: run `using LaTeXStrings` before plotting.")
export texlabel
