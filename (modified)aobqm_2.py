import numpy as np
import matplotlib.pyplot as plt
import time

def quantize_data(data, num_bits=8):
    """
    Quantizes the input data using integer representation.
    
    Args:
        data (ndarray): Input data to be quantized.
        num_bits (int): The number of bits for quantization (default: 8).

    Returns:
        ndarray: Quantized data.
    """
    # Calculate the scaling factor based on the range of the data
    min_val = np.min(data)
    max_val = np.max(data)
    scale = (2 ** num_bits - 1) / (max_val - min_val)
    
    # Quantize the data
    quantized_data = np.round((data - min_val) * scale).astype(int)
    
    return quantized_data, scale, min_val


def reconstruct_data(quantized_data, scale, min_val, num_bits=8):
    """
    Reconstructs the data from quantized values.
    
    Args:
        quantized_data (ndarray): Quantized data.
        scale (float): The scaling factor used in quantization.
        min_val (float): The minimum value used for normalization.
        num_bits (int): The number of bits for quantization (default: 8).
    
    Returns:
        ndarray: Reconstructed data.
    """
    # Reconstruct the data
    reconstructed_data = (quantized_data / (2 ** num_bits - 1)) * (np.max(data) - min_val) + min_val
    return reconstructed_data


def calculate_performance(original_data, reconstructed_data):
    """
    Calculate MSE and PSNR between the original and reconstructed data.
    
    Args:
        original_data (ndarray): The original data.
        reconstructed_data (ndarray): The reconstructed data.
        
    Returns:
        tuple: MSE (Mean Squared Error) and PSNR (Peak Signal-to-Noise Ratio).
    """
    mse = np.mean((original_data - reconstructed_data) ** 2)
    psnr = 20 * np.log10(np.max(original_data) / np.sqrt(mse))
    
    return mse, psnr


def plot_results(original_data, reconstructed_data, error_map):
    """
    Plot the original data, reconstructed data, and error map.
    
    Args:
        original_data (ndarray): The original data.
        reconstructed_data (ndarray): The reconstructed data.
        error_map (ndarray): The error map between the original and reconstructed data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the original data
    cax1 = axes[0].imshow(original_data, cmap='viridis')
    axes[0].set_title('Original Data')
    fig.colorbar(cax1, ax=axes[0])
    
    # Plot the reconstructed data
    cax2 = axes[1].imshow(reconstructed_data, cmap='viridis')
    axes[1].set_title('Reconstructed Data')
    fig.colorbar(cax2, ax=axes[1])
    
    # Plot the error map
    cax3 = axes[2].imshow(error_map, cmap='coolwarm')
    axes[2].set_title('Error Map')
    fig.colorbar(cax3, ax=axes[2])
    
    plt.show()


def compute_execution_time(func, *args):
    """
    Measure the execution time of a function.
    
    Args:
        func (function): The function whose execution time will be measured.
        *args: Arguments to pass to the function.
        
    Returns:
        float: Execution time in seconds.
    """
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time


# Example usage
if __name__ == "__main__":
    # The user needs to supply their own data (e.g., 2D NumPy array)
    data = np.load('data.npy')  # Load your own data (replace 'data.npy' with your file path)
    
    # Perform quantization and reconstruction
    quantized_data, scale, min_val = quantize_data(data, num_bits=8)
    reconstructed_data = reconstruct_data(quantized_data, scale, min_val, num_bits=8)
    
    # Calculate performance
    mse, psnr = calculate_performance(data, reconstructed_data)
    
    # Generate error map
    error_map = reconstructed_data - data
    
    # Plot results
    plot_results(data, reconstructed_data, error_map)
    
    # Print results
    print(f"MSE: {mse}")
    print(f"PSNR: {psnr} dB")
    
    # Measure execution time
    quantization_time = compute_execution_time(quantize_data, data, 8)
    reconstruction_time = compute_execution_time(reconstruct_data, quantized_data, scale, min_val, 8)
    
    print(f"Quantization Time: {quantization_time} seconds")
    print(f"Reconstruction Time: {reconstruction_time} seconds")
