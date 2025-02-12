import numpy as np
import matplotlib.pyplot as plt


def plot_mask(filepath, savepath):
    binary_array = np.load(filepath)
    grid_shape = binary_array.shape
    
    fig, ax = plt.subplots(figsize=(grid_shape[1], grid_shape[0]))
    ax.imshow(binary_array, cmap='gray_r', vmin=0, vmax=1)  
    
    # Draw grid lines
    for i in range(grid_shape[0] + 1):
        ax.axhline(i - 0.5, color='grey', linewidth=1)
    for j in range(grid_shape[1] + 1):
        ax.axvline(j - 0.5, color='grey', linewidth=1)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(savepath)




def plot_loss(filepath, savepath):
    name = ["mask", "distill", "ppl"]
    
    data = np.load(filepath)
    t, c = data.shape

    x = np.arange(t)
    plt.figure(figsize=(15, 10))
    for i in range(c):
        plt.plot(x, data[:, i], label=name[i])
    plt.ylim(0, 4)
    plt.legend(fontsize = 16)
    plt.savefig(savepath)
    
    
    
    


import os

def main():
    target_path = "weight/dyllm_test_5"
    
    mask_path = os.path.join(target_path, "token_mask.npy")
    loss_path = os.path.join(target_path, "lossfile.npy")
    
    save_mask = os.path.join(target_path, "token_mask.png")
    loss_save = os.path.join(target_path, "loss_graph.png")
    
    
    
    plot_mask(mask_path, save_mask)
    plot_loss(loss_path, loss_save)
    
    
    
    
    
    
    
    return



if __name__ == "__main__":
    main()