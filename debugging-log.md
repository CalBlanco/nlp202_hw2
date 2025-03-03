## MPS Device Allocation Issue

### Attempts:
1. ✓ Added proper device checking
2. ✓ Explicit device movement for model and data

### Key Points:
- Always move both model AND data to device
- Check device availability before assignment
- Ensure consistent device usage across all operations 

Issue: Loss still not decreasing
- Root cause: Likely initialization or optimization issues

Radical Solutions:
1. Complete model reset with proper weight initialization (Xavier)
2. Zero-initialized transition matrix instead of random
3. Added dropout to LSTM (0.5)
4. Smaller batch size (16)
5. Larger embedding and hidden dimensions
6. Tighter gradient clipping (0.5)
7. Simple SGD with small learning rate
8. More training epochs (50)
9. Best model tracking 

# BiLSTM Type Mismatch Debug - [Date]

## Issue
- LSTM layer receiving incorrect tensor type (int64 instead of float32)

## Attempt 1 ✅
- Re-enabled embedding layer to handle type conversion
- Result: Success - Embedding layer now converts int64 indices to float32 embeddings

## Root Cause
Input tensors were bypassing the embedding layer, causing type mismatch at LSTM layer 