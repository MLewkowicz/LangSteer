#!/usr/bin/env python3
"""DEPRECATED: CALVIN to Zarr conversion script.

This script has been moved to training/policies/dp3/preprocessing/convert_calvin.py

Please use:
    python -m training.policies.dp3.preprocessing.convert_calvin

This wrapper will be removed in a future version.
"""

import warnings
warnings.warn(
    "convert_calvin_to_zarr.py is deprecated. "
    "Use: python -m training.policies.dp3.preprocessing.convert_calvin",
    DeprecationWarning,
    stacklevel=2
)

print("\nDEPRECATED: This script has moved to training/policies/dp3/preprocessing/")
print("Please use: python -m training.policies.dp3.preprocessing.convert_calvin\n")
