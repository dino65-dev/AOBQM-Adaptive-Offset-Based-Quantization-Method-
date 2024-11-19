import numpy as np

float_array = # Load your data here
# Adaptive scaling factor for high precision and minimal overflow
adaptive_scaling_factor = 1_000_000_000
offset_adjustment = 288.5  # Adjust based on your dataset's distribution

def quantize_with_adaptive_scaling(data, scaling_factor=adaptive_scaling_factor, offset=offset_adjustment):
    # Get absolute values and signs
    sign_array = np.sign(data).astype(np.int8)
    absolute_values = np.abs(data)

    # Integer and decimal quantization with adaptive scaling
    integer_part = np.floor(absolute_values).astype(np.uint8)
    decimal_part = np.round((absolute_values - integer_part) * scaling_factor).astype(np.int64)  # int64 for large values

    # Offset quantization for higher precision
    offset_adjusted_values = data + offset
    integer_offset_part = np.floor(offset_adjusted_values).astype(np.int8)
    decimal_offset_part = np.round((offset_adjusted_values - integer_offset_part) * scaling_factor).astype(np.int64)

    # Reconstruct values
    reconstructed_values = sign_array * (integer_part + decimal_part / scaling_factor)
    reconstructed_values_offset = integer_offset_part + decimal_offset_part / scaling_factor - offset

    # Calculate reconstruction errors
    reconstruction_error = data - reconstructed_values
    reconstruction_error_offset = data - reconstructed_values_offset

    return {
        "sign_array": sign_array,
        "integer_part": integer_part,
        "decimal_part": decimal_part,
        "integer_offset_part": integer_offset_part,
        "decimal_offset_part": decimal_offset_part,
        "reconstructed_values": reconstructed_values,
        "reconstruction_error": reconstruction_error,
        "reconstructed_values_offset": reconstructed_values_offset,
        "reconstruction_error_offset": reconstruction_error_offset
    }

# Sample test data
data = float_array     #np.random.rand(100, 100)  # Replace with your dataset

# Run the quantization process
results = quantize_with_adaptive_scaling(data)

# Display results for analysis
print("Adaptive Decimal Scaling Factor:", adaptive_scaling_factor)
print("Sign Array:", results["sign_array"])
print("Integer Part (uint8):", results["integer_part"])
print("Decimal Part (int64):", results["decimal_part"])
print("Offset for Offset Quantization:", offset_adjustment)
print("Integer Offset Part (int8):", results["integer_offset_part"])
print("Decimal Offset Part (int64):", results["decimal_offset_part"])
print("\nReconstructed Values with Adaptive Scaling:", results["reconstructed_values"])
print("Reconstruction Error with Adaptive Scaling:", results["reconstruction_error"])
print("\nReconstructed Values with Offset Quantization:", results["reconstructed_values_offset"])
print("Reconstruction Error with Offset Quantization:", results["reconstruction_error_offset"])
