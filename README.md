# MNIST Classification using Graph Neural Networks (GNNs)

This project demonstrates an unconventional yet powerful approach to image classification by representing MNIST handwritten digits as graphs and applying a Graph Convolutional Network (GCN) to classify them. It serves as a complete pipeline from data preprocessing and graph creation to model training, evaluation, and visualization.

## Key Features
- **Image-to-Graph Conversion:** Converts standard pixel-based images into graph structures using the SLIC superpixel algorithm.
- **GCN Implementation:** A Graph Convolutional Network built with PyTorch and the PyTorch Geometric (PyG) library.
- **Detailed Evaluation:** Generates comprehensive performance metrics, including accuracy, a per-class classification report, and a visual confusion matrix.
- **Modular Scripts:** The project is organized into clear, sequential scripts for data processing, training, and testing.

## Methodology
Instead of treating an image as a grid of pixels, we first transform it into a graph. Each image is segmented into approximately 75 **superpixels**—small, perceptually uniform regions.

- **Nodes:** Each superpixel becomes a node in the graph. The node's features include its average pixel intensity and its normalized (x, y) coordinates.
- **Edges:** An edge is created between two nodes if their corresponding superpixels are adjacent in the original image.

This process turns each 28x28 image into a graph that the GNN can process.

![Methodology Diagram](https://i.imgur.com/z1hXGxE.png)

## Project Structure
```
.
├── mnist_graphs/
│   └── processed/
│       ├── data_0.pt
│       ├── data_1.pt
│       └── ...
├── gnn_mnist_model.pth
├── process_data.py
├── train_gnn.py
└── test.py
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install torch torchvision torch-geometric scikit-image scikit-learn matplotlib seaborn numpy tqdm
    ```

## Usage
Follow these steps in order to run the complete pipeline.

### Step 1: Process the Data
First, run the data processing script. This will download the MNIST dataset, convert all 60,000 training images into graph objects, and save them in the `mnist_graphs/processed/` directory.

```bash
python process_data.py
```

### Step 2: Train the Model
Next, train the GNN on the newly created graph dataset. This script will train the model, display performance metrics for each epoch, and save the final model weights to `gnn_mnist_model.pth`.

```bash
python train_gnn.py
```

### Step 3: Evaluate the Model
Finally, run the test script to get a detailed performance analysis of your trained model on the test set.

```bash
python test.py
```
This script will print a classification report and a confusion matrix to the console, and it will also display plots for the confusion matrix and sample predictions.

## Results
The baseline model achieves an accuracy of approximately **84%**. While not state-of-the-art for the MNIST problem, it serves as a strong proof-of-concept for the graph-based classification method.

The evaluation script generates detailed visualizations of model performance, such as:

*(Here you can add screenshots of your own generated plots, like the confusion matrix or the sample predictions grid.)*
<img width="1362" height="640" alt="image" src="https://github.com/user-attachments/assets/1581c103-e5cb-4b96-a4a3-2802284cad9a" />

## Future Improvements
The model's performance can be significantly improved by(currently working on):
- **Enhancing the model architecture** by making the layers wider or using more advanced GNN layers like `GATConv`.
- **Adding normalization layers** (e.g., `BatchNorm`) between GNN layers.
- **Tuning the graph creation process** by experimenting with a different number of superpixels in `process_data.py`.
