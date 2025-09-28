import matplotlib.pyplot as plt

# 1. Data from your output
epochs = range(1, 13)
train_accuracy = [0.8296, 0.8839, 0.8987, 0.9098, 0.9159, 0.9225, 0.9284, 0.9333, 0.9387, 0.9436, 0.9449, 0.9503]
val_accuracy = [0.7312, 0.8970, 0.8750, 0.9032, 0.9047, 0.9132, 0.9095, 0.9135, 0.9165, 0.9112, 0.9048, 0.9128]

# 2. Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')

# 3. Add labels and title
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 4. Save the plot to a file
plt.savefig('accuracy_graph.png')

print("Graph saved as accuracy_graph.png")