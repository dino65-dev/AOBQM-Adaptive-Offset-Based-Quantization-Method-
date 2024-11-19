import numpy as np

def adaptive_quantize(data, scaling_factor=None, offset=None):
    """
    Quantizes floating-point data using adaptive scaling and offset quantization.
    
    Parameters:
    - data (ndarray): Input array of floating-point values to quantize.
    - scaling_factor (int, optional): Scaling factor for the decimal part. If None, an adaptive factor is used.
    - offset (float, optional): Offset for the integer part. If None, it will be computed as the median of the data.
    
    Returns:
    - quantized_data (dict): Dictionary containing quantized components and reconstruction information:
        - "sign_array": Sign array for each element (-1 or 1).
        - "integer_part": Unsigned integer part (uint8).
        - "decimal_part": Scaled integer decimal part (int64).
        - "integer_offset_part": Offset-adjusted integer part (int8).
        - "decimal_offset_part": Offset-adjusted decimal part (int64).
        - "scaling_factor": Scaling factor used for quantization.
        - "offset": Offset used for integer quantization.
        - "reconstructed_values": Reconstructed values using adaptive scaling.
        - "reconstructed_values_offset": Reconstructed values using offset quantization.
        - "reconstruction_error": Error between original and reconstructed values.
        - "reconstruction_error_offset": Error for offset-based reconstruction.
    """
    
    # Ensure data is in floating point format for proper quantization
    data = np.asarray(data, dtype=np.float32)
    
    # Step 1: Set adaptive scaling factor if not provided
    if scaling_factor is None:
        scaling_factor = 10 ** np.ceil(-np.log10(np.min(np.abs(data[data != 0]))))
    
    # Step 2: Define sign array to store the sign of the original data
    sign_array = np.sign(data).astype(int)
    
    # Step 3: Separate integer and decimal parts with adaptive scaling
    integer_part = np.floor(np.abs(data)).astype(np.uint8)
    decimal_part = np.round((np.abs(data) - integer_part) * scaling_factor).astype(np.int64)
    
    # Step 4: Compute offset as median of data if not provided
    if offset is None:
        offset = np.median(integer_part)
    
    # Step 5: Perform offset quantization on integer and decimal parts
    integer_offset_part = (integer_part - offset).astype(np.int8)
    decimal_offset_part = np.round((np.abs(data) - integer_offset_part - offset) * scaling_factor).astype(np.int64)
    
    # Step 6: Reconstruct original values from quantized components
    reconstructed_values = (sign_array * (integer_part + decimal_part / scaling_factor)).astype(np.float32)
    reconstructed_values_offset = (sign_array * (integer_offset_part + offset + decimal_offset_part / scaling_factor)).astype(np.float32)
    
    # Step 7: Calculate reconstruction errors
    reconstruction_error = data - reconstructed_values
    reconstruction_error_offset = data - reconstructed_values_offset
    
    # Return all quantized components and reconstruction information
    return {
        "sign_array": sign_array,
        "integer_part": integer_part,
        "decimal_part": decimal_part,
        "integer_offset_part": integer_offset_part,
        "decimal_offset_part": decimal_offset_part,
        "scaling_factor": scaling_factor,
        "offset": offset,
        "reconstructed_values": reconstructed_values,
        "reconstructed_values_offset": reconstructed_values_offset,
        "reconstruction_error": reconstruction_error,
        "reconstruction_error_offset": reconstruction_error_offset
    }

# Example usage
data = np.random.rand(10, 10) - 0.5  # Random data between -0.5 and 0.5
quantized_data = adaptive_quantize(data)
print(quantized_data)
