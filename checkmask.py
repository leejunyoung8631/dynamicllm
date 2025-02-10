import numpy as np
import matplotlib.pyplot as plt



def plot_binary_grid_with_boundaries(binary_list, grid_shape, savefile):
    binary_array = np.array(binary_list).reshape(grid_shape)
    
    fig, ax = plt.subplots(figsize=(grid_shape[1], grid_shape[0]))
    ax.imshow(binary_array, cmap='gray_r', vmin=0, vmax=1)  # 'gray_r' makes 0 black and 1 white
    
    # Draw grid lines
    for i in range(grid_shape[0] + 1):
        ax.axhline(i - 0.5, color='grey', linewidth=1)
    for j in range(grid_shape[1] + 1):
        ax.axvline(j - 0.5, color='grey', linewidth=1)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(savefile)






def main():
    savefile = "./mask.png"
    mask = np.load("./weight/dyllm_test_5/token_mask.npy")
    print(mask.shape)
    print(mask[0])
    
    binary_list = mask
    grid_shape = binary_list.shape
    plot_binary_grid_with_boundaries(binary_list, grid_shape, savefile)
    
    
    return



if __name__ == "__main__":
    main()
