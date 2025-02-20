## MPS Device Allocation Issue

### Attempts:
1. ✓ Added proper device checking
2. ✓ Explicit device movement for model and data

### Key Points:
- Always move both model AND data to device
- Check device availability before assignment
- Ensure consistent device usage across all operations 