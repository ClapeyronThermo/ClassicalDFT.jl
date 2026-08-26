```@meta
CurrentModule = ClassicalDFT
```

## Contents

```@contents
Pages = ["free_energy.md"]
Depth = 1
```

## Free Energy Evaluation

Low-level functions used internally by [`converge!`](@ref ClassicalDFT.converge!) and the
[Methods](methods.md) to evaluate the free-energy functional and its density derivative
for a given density profile. Not typically called directly by users.

```@docs
ClassicalDFT.free_energy
ClassicalDFT.grand_potential
ClassicalDFT.δFδρ_res
ClassicalDFT.F_res
ClassicalDFT.F_ideal
```
