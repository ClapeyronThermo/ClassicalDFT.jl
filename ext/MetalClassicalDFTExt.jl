module MetalClassicalDFTExt

using ClassicalDFT
using Metal

# Enzyme forward-mode-differentiated kernels (both :forward and :forward_batch) crash
# Metal's AGX shader compiler ("Compiler encountered an internal error (AGXMetal..., code
# 3)") for non-trivial free-energy terms (e.g. any SAFT-family dispersion term), while
# Enzyme reverse-mode compiles and runs correctly for the identical math. Default to
# :reverse on Metal until that's fixed upstream; users can still opt into :forward /
# :forward_batch explicitly (e.g. for benchmarking) via DFTOptions(device, ad_mode).
ClassicalDFT.DFTOptions(::Metal.MetalBackend) = ClassicalDFT.DFTOptions(MetalBackend(); ad_mode = :reverse, precision = Float32)

end
