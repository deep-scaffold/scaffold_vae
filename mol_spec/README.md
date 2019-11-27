# Molecule specification

A utility module for managing atom types and bond types. Shared by all molecule
generative models.

Usage:
```python
from mol_spec import MoleculeSpec

# Getting the default molecule specification from the package
ms = MoleculeSpec.get_default()
```