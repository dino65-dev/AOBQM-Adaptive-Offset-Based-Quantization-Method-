# AOBQM [Adaptive-Offset-Based-Quantization-Method](http://dx.doi.org/10.13140/RG.2.2.31924.87688)
# Quantization Method for Neural Networks using Adaptive Scaling and Offset Quantization
## Overview
This repository presents a novel quantization method that separates a float16 value into two int8 arrays: one for the integer part and one for the decimal part. This approach allows the reconstruction of the original float16 value by combining the integer and decimal parts. The method also features adaptive scaling and offset quantization to improve accuracy and reduce reconstruction error.

This technique provides an efficient way to quantize floating-point values and is applicable in machine learning models, particularly for reducing storage requirements and improving inference performance without significant loss in precision.

## Citation

If you use this software, please cite the following:

```bibtex
@article{https://doi.org/10.13140/rg.2.2.31924.87688,
  doi = {10.13140/RG.2.2.31924.87688},
  url = {https://rgdoi.net/10.13140/RG.2.2.31924.87688},
  author = {{Dinmay Brahma}},
  language = {en},
  title = {An Adaptive Offset-Based Quantization Method Using Separate Integer and Decimal Representations for Low-Precision Computations in Edge AI},
  publisher = {Unpublished},
  year = {2024}
}
```

## Key Features
**Integer-Decimal Split:** Separates the integer and decimal components of the float16 values for independent quantization.
**Adaptive Decimal Scaling:** Dynamically scales the decimal part based on the values to ensure more accurate representation.
**Offset Quantization:** Introduces an offset to handle negative numbers and improve reconstruction.
**Minimal Reconstruction Error:** The method demonstrates minimal reconstruction error, ensuring that quantized values closely match the original values.
## Methodology
**Quantization:** The floating-point number is split into two parts:

The integer part is stored using 8-bit integer representation (int8).
The decimal part is scaled to an appropriate range using an adaptive scaling factor and is stored as a 64-bit integer (int64).
**Offset Quantization:** An offset is introduced to account for negative values. This is calculated by determining the minimum possible value in the data and adjusting the quantization range accordingly.

**Reconstruction:** To reconstruct the original values, the integer and decimal parts are combined using their respective scaling factors.

## Reconstruction Error
The proposed method ensures minimal reconstruction error, making the quantized values almost identical to the original float16 values. The method adapts the decimal scaling and offset quantization dynamically based on the data, improving the accuracy of the reconstruction.


# Quantization
integer_part, decimal_part, integer_offset_part, decimal_offset_part, scaling_factor, offset = quantize(data)

# Reconstruction
reconstructed_values, reconstructed_offset_values = reconstruct(integer_part, decimal_part, integer_offset_part, decimal_offset_part, scaling_factor, offset)

# Print results
print("Original Data:", data)
print("Reconstructed Data (Adaptive Scaling):", reconstructed_values)
print("Reconstructed Data (Offset Quantization):", reconstructed_offset_values)
```
## Example Results
**Original Data:** Floating-point values ranging from -1.0 to 1.0.
**Reconstructed Data (Adaptive Scaling):** Values reconstructed with minimal error, closely matching the original data.
**Reconstructed Data (Offset Quantization):** Values reconstructed using offset quantization with negligible error.

## Performance
The quantization method demonstrates excellent performance in terms of both reconstruction accuracy and computational efficiency. The adaptive scaling and offset quantization allow the method to maintain precision while significantly reducing storage space.

## Applications
**Machine Learning:** Can be used to quantize weights and activations in deep neural networks, reducing memory usage and improving inference speed.
**Embedded Systems:** Suitable for use in resource-constrained environments where memory is limited.
**Data Compression:** Effective in scenarios where data needs to be compressed without significant loss of precision.
## Future Work
**Future improvements may involve:**

Further optimizations in the scaling factor selection.
Evaluation in different machine learning tasks to measure the real-world impact.
Enhancements to handle larger datasets efficiently.

## Contact
**For questions or contributions, please contact [Dinmay Kumar Brahma] at [dinmaybrahma@outlook.com].**

## Licensing
This project is licensed under the ```GPL-3.0```.
