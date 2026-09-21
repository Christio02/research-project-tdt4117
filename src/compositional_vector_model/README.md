# Compositional Vector Model

Implementation of composition functions for distributional semantic vectors,
per Mitchell & Lapata (2010).

## Contents

- `composition.py` — vector composition functions:
  - additive: p = u + v
  - weighted additive: p = a*u + b*v
  - multiplicative: p_i = u_i * v_i (element-wise)
  - dilation: stretch v along u by factor λ
  - `compose_query`: compose multi-term queries using any of the above methods

## Reference

- Mitchell, J., & Lapata, M. (2010). *Composition in Distributional Models of
  Semantics.* Cognitive Science, 34, 1388–1429.
