module LaTeXStringscDFTExt

using ClassicalDFT
using LaTeXStrings

const _UNICODE_TO_LATEX = (
    "ρ" => "\\rho",
    "σ" => "\\sigma",
    "ω" => "\\omega",
    "Å" => "\\mathring{A}",
    "³" => "^3",
    "²" => "^2",
)

# MathTeXEngine (Makie's LaTeX renderer) always typesets a plain ASCII '-' in math mode as a
# widely-spaced minus sign (e.g. "PC-SAFT" -> "PC − SAFT"), regardless of surrounding font
# commands like \mathrm{}. Swapping in the Unicode hyphen U+2010 sidesteps that special-cased
# substitution entirely (it isn't recognized as the minus character), so dashes in
# species/model names and compound units stay tight, ordinary hyphens.
function _texbody(s::AbstractString)
    out = s
    for (u, l) in _UNICODE_TO_LATEX
        out = replace(out, u => l)
    end
    return replace(out, "-" => "‐")
end

function ClassicalDFT.texlabel(s::String; upright::Bool=false)
    if upright
        # \; forces a visible math-mode space -- a literal space inside \mathrm{} is not
        # guaranteed to survive MathTeXEngine's parser the way it would in real LaTeX.
        out = "\\mathrm{" * replace(_texbody(s), " " => "\\;") * "}"
        return LaTeXStrings.latexstring(out)
    end

    # Axis labels follow the "quantity / unit" convention: the quantity symbol is left in the
    # default italic math font, while the unit (everything after " / ") is set upright, since
    # units are not italicized in scientific typesetting.
    parts = split(s, " / "; limit=2)
    out = if length(parts) == 2
        symbol, unit = parts
        _texbody(symbol) * " / \\mathrm{" * replace(_texbody(unit), " " => "\\;") * "}"
    else
        _texbody(s)
    end
    return LaTeXStrings.latexstring(out)
end

end
