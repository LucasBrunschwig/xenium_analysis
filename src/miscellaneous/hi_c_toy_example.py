import numpy as np
import matplotlib.pyplot as plt

# Create a toy Hi-C matrix
hi_c_matrix = np.array([
    [0, 5, 2, 1],
    [5, 0, 3, 0],
    [2, 3, 0, 4],
    [1, 0, 4, 0]
])

# Normalize the matrix (optional for illustration purposes)
normalized_matrix = hi_c_matrix / hi_c_matrix.sum(axis=0)

# Perform PCA
pca_result = np.linalg.svd(normalized_matrix, full_matrices=False)
pc1 = pca_result[2][0]  # Extract PC1 values

# Assign A/B compartments based on PC1 values
compartment_assignment = np.sign(pc1)

# Visualize the Hi-C matrix and A/B compartments
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the Hi-C matrix
cax1 = ax1.matshow(normalized_matrix, cmap='viridis')
ax1.set_title('Normalized Hi-C Matrix')
fig.colorbar(cax1, ax=ax1)

# Plot A/B compartments
compartment_colors = ['red' if compartment == -1 else 'blue' for compartment in compartment_assignment]
cax2 = ax2.matshow(np.array([compartment_assignment]), cmap='coolwarm', aspect='auto')
ax2.set_title('A/B Compartments')
ax2.set_yticks([])  # Remove y-axis ticks
ax2.set_xticks(range(len(compartment_assignment)))
ax2.set_xticklabels(['A', 'B', 'C', 'D'])

# Add color legend
legend_labels = ['B Compartment', 'A Compartment']
legend_colors = ['red', 'blue']
legend_patches = [plt.Line2D([0], [0], marker='o', color=color, label=label, linestyle='None') for color, label in zip(legend_colors, legend_labels)]
ax2.legend(handles=legend_patches, loc='upper right')

plt.show()