# GATv2 Node Classification – CUDA Implementation

# Introduction

This repository provides a Graph Attention Network (GATv2) implementation in CUDA/C++ for single-label node classification on graph-structured data. GATv2 is an improved version of the original GAT model, introducing a more expressive dynamic attention mechanism (as opposed to static attention in GAT)

Ref-1: https://arxiv.org/abs/2105.14491#:~:text=representation%20learning%20with%20graphs,a%20simple%20fix%20by%20modifying

Ref-2: https://arxiv.org/abs/1710.10903  
This implementation is built from scratch without external deep learning libraries. It includes the full training pipeline (forward pass, backpropagation, and parameter updates) on the GPU. The code supports multiple GATv2 layers, multi-head attention, and both SGD and Adam optimizers, with optional gradient clipping for stability.

# Key Features
- **Choosing an Implementation (Node vs Edge):** Both implementations produce the same results, but use different parallelization strategies:
   - _Node-centric (Thread-block per node):_ Each node's neighborhood is processed by one thread block. This approach may be beneficial if the graph has a fairly uniform and not too large degree distribution, so that each block has a reasonable amount of work and memory. Very high-degree nodes could lead to larger loops within a single block.
   - _Edge-centric (Thread per edge):_ Each GPU thread handles computations for a single edge (source–destination pair) in the graph. All edges are processed in parallel. This can better utilize the GPU for very large graphs with many edges, as it maximizes parallel thread count. However, since multiple threads will write to the same destination node's accumulator, atomic operations are used for safely accumulating results (e.g., atomic adds when summing messages from edges to a node). This can incur overhead, especially if there are many threads contending on the same node.
 - **Configurable Multi-Layer GATv2:** conveniently set the number of layers, number of attention heads per layer, and output dimensions per layer via command-line arguments.
 - **Graph Data Handling:** The code reads input graphs in CSR format (with separate four txt files for features, graph structure, and labels) and converts to COO internally for edge-wise operations.
 - **Training on GPU:** All heavy computations (attention score calculation, feature aggregation, loss and gradient computations, and weight updates) are done on the GPU for efficiency. The program prints training progress (loss and accuracy) each epoch and supports monitoring GPU memory usage.

 # Dataset Preparation:
 To run the code, you will need to provide a graph dataset in the particular CSR format. Some standard datasets are available in this Google drive link https://drive.google.com/drive/folders/1Ubam1XbIzbHyoVx3j30NOhbbh9xbjR6i?usp=share_linkDownload .  Download the datasets from the provided Google Drive folder and extract them under a root data directory. Each dataset should reside in its own subdirectory within the data root, containing the following files:
- **features.txt** – Node feature matrix (one line per node, features separated by spaces).
-    **row_ptr.txt**  – CSR row pointer array of length (N+1), where N is number of nodes.  `row_ptr[i]`  gives the index in  `col_idx.txt`  where the adjacency list of node  _i_  starts.
-  **col_idx.txt** – CSR column index array of length E (number of edges). It lists all neighbor node indices for each node.
- **labels.txt** – Node labels (one integer class label per line, for each node).

Make sure the folder name matches the dataset name you will use (all lowercase, e.g., `cora`, `citeseer`, etc.). You can place these folders in a default location (`./data` by default) or specify a custom path with `--data-root`. If you do not use the `--data-root` flag, the code will also check the `DATA_ROOT` environment variable for a default path.

### Datasets available in the download folder:
- **Cora** – Citation network of scientific publications in machine learning. _N_ = 2,708 nodes, _E_ = 5,429 edges, 7 classes, 1,433 features per node. 
- **Citeseer** – Citation network of research papers. _N_ = 3,327 nodes, _E_ = 4,732 edges, 6 classes, 3,703 features per node.
- **Pubmed** – Citation network of medical research papers. _N_ = 19,717 nodes, _E_ = 44,338 edges, 3 classes, 500 features per node.
- **Amazon Products** (OGBN-Products) – Amazon product co-purchase network from OGB. _N_ ≈ 2.45 million products, _E_ ≈ 61.9 million edges, 47 product categories (classes). Each node has a 100-dimensional feature vector derived from product text descriptions.
- **OGBN-Arxiv** – Citation network of ArXiv CS papers (from OGB). _N_ = 169,343 papers, _E_ = 1,166,243 directed citation edges, 40 subject-area classes. Each node’s features are a 128-dim bag-of-words embedding of its title and abstract.
**Note:** Cora, Citeseer, and Pubmed are small citation benchmarks commonly used in GNN research. OGBN-Arxiv and OGBN-Products are larger Open Graph Benchmark datasets.

## Requirements and Setup:

-   **CUDA Toolkit:**   You need a CUDA-capable NVIDIA GPU and the CUDA toolkit installed. The code is compatible with C++11 and requires CUDA 9.0 or later due to the use of warp shuffle sync operations. For optimal performance, CUDA 11+ is recommended.
    
-   **Compute Capability:**  The example compilation uses  `-arch=sm_75`  (NVIDIA Turing architecture, compute 7.5). You may change this to match your GPU (e.g.,  `sm_70`,  `sm_80`,  `sm_86`, etc.).
    
-   **Memory:**  GPU memory usage depends on the dataset and model size (number of layers and dimensions). The program will print memory info before and after allocation. For reference, Cora/Citeseer are very small (fit in under 100MB), Pubmed is moderate (~ hundreds of MB), while OGBN-products can occupy few GB of memory.
    
-   **C++ Standard Library:**  Uses  `<iostream>`,  `<vector>`, etc., which are included by NVCC by default. Also uses Thrust (for some reductions), which comes with CUDA.
No additional dependencies are required beyond the CUDA runtime.

## Compilation
Clone or download this repository to your local machine or coding environment (e.g., Kaggle Notebook or Google Colab). Then compile the CUDA code using  `nvcc`. We provide two source files corresponding to the two implementation variants:

-   **`GATv2_node_based.cu`**  – Node-centric implementation (thread block per node).
-   **`GATv2_edge_based.cu`**  – Edge-centric implementation (thread per edge). 

You can compile one or both of them. For example, to compile the node-centric version:
    `nvcc -std=c++17 -arch=sm_75 GATv2_node_based.cu -o train_node `

Similarly, for the edge-centric version: 
 `nvcc -std=c++17 -arch=sm_75 GATv2_edge_based.cu -o train_edge`

This will produce executables `train_node` and `train_edge` respectively. Make sure to include the correct path to the `.cu`file if you are not in the same directory.
 
 - **Note:** If you are running in a Jupyter environment (like Kaggle kernels), prefix the commands with `!` to run shell commands. For example, in a notebook cell:
 `!nvcc -std=c++17 -arch=sm_75 GATv2_node_based.cu -o train_node`
`!./train_node --num-layers 3`

####  Running Environment:
1. Local Machine:
   - Remove the first line `"%%writefile cuda.cu"` when running locally
   - Compile directly with nvcc

2. Google Colab/Kaggle:
   - Keep the first line `"%%writefile cuda.cu"`
   - This magic command will create a .cu file in your notebook environment
   - This is necessary because Colab/Kaggle use Jupyter notebooks and cannot compile CUDA code directly
   - After running the cell with the code, you'll need to compile the generated .cu file separately

Example for Colab/Kaggle workflow:
1. Run the cell with the code (including `%%writefile cuda.cu)`
2. Compile using: `!nvcc cuda.cu -o cuda_program`
3. Run using: `!./cuda_program`

## Usage Instructions
After compiling, you can run the training program with various command-line options to configure the model and training hyperparameters. **Both  `train_node`  and  `train_edge`  accept the same arguments.** The primary difference is which kernel implementation they use internally. Both will train a GATv2 model on the specified dataset.

### Available CLI Options:
- `--num-layers L` – **(Integer)** Number of GATv2 layers in the model (not counting the final classification layer). **Default:**2.
_Usage:_ If you specify `L`, you must also provide exactly `L` comma-separated values for `--heads` and `--outdims` options (see below).
- `--heads h1,h2,...,hL` – **(List of integers)** Number of attention heads for each GATv2 layer. The list must have length equal to `--num-layers`. For example, `--num-layers 3 --heads 4,1,1` means 3 layers with 4 heads in the first layer, 1 head in second, 1 head in third.  
_Note:_ More heads in a layer means that layer's output features are the concatenation of outputs from each head. For intermediate layers, the next layer's input dimension will be `heads[layer] * outdim[layer]` (since outputs of all heads are concatenated).
- `--outdims d1,d2,...,dL`  –  **(List of integers)**  Output dimension for  **each head**  at each layer. Provide a comma-separated list of length  `L`. For example, with  `--heads 4,1,1 --outdims 64,32,16`:

  -  Layer1 has 4 heads, each head produces a 64-dimensional output (so layer1 outputs 4×64 = 256 features per node).
  -   Layer2 has 1 head producing 32-dimensional output.
  -   Layer3 has 1 head producing 16-dimensional output (final embedding size per node before the classifier).  
    The final classification layer (built into the code) will take the last layer's output and produce logits for each class.
-  `--epochs N` – **(Integer)** Number of training epochs. **Default:** 200.
- `--optimizer OPT` – **(String)** Optimizer to use: `"sgd"` or `"adam"`. **Default:**  `"sgd"`.
- If `adam` is chosen, you can adjust the beta parameters (see below). If `sgd` is chosen, any provided beta parameters will be ignored (and a warning will be shown).
- `--beta1 B1`, `--beta2 B2` – **(Float)** Beta parameters for Adam optimizer (β1 and β2). **Default:** 0.9 and 0.999. These are ignored if optimizer is SGD. (They must be in the range (0,1) if specified when using Adam, otherwise the program will error out.)
- `--lr α` – **(Float)** Learning rate. **Default:** 0.0001 (1e-4). You will likely want to increase this (e.g., 0.01) for quicker training, especially with SGD.
- `--clip` – **(Flag)** Enable gradient clipping. If this flag is present, the program will clip the norm of the gradients of each trainable parameter group to a fixed threshold (5.0 by default in code). This can help stabilize training if gradients explode. Gradient clipping is off by default, but you can turn it on by simply adding `--clip` (no value needed).
- `--dataset NAME` – **(String)** Name of the dataset to train on. **Default:**  `"pubmed"`. Supported names are `cora`, `citeseer`, `pubmed`, `amazon-products` (for OGBN-Products), `ogbn-arxiv` (and any others you have prepared in the same format). The program will look for a subdirectory with this name under the data root, containing the required .txt files. If the dataset is not found or files are missing, it will error.
- `--data-root PATH` – **(String)** Path to the root directory containing dataset folders. **Default:**  `"./data"` (current directory under a folder named data). You can use this flag if your data is located elsewhere. For example, `--data-root /kaggle/input` (as in the Kaggle example below) or `--data-root C:\Graphs\`. The path can be absolute or relative. The code ensures a trailing slash is added internally.

### Example Command:
Suppose we want to run a 3-layer GATv2 on the Citeseer dataset, with 4 attention heads in the first layer and 1 head in each subsequent layer. Let each head output 64 features in layer1, 32 in layer2, and 16 in layer3. We choose Adam optimizer with learning rate 0.01, and enable gradient clipping. We assume the Citeseer data is stored in `/data/graphs/citeseer/` (with the four .txt files inside). The command would be:  
  `./train_node --num-layers 3  --heads 4,1,1 --outdims 64,32,16 --epochs 200  --optimizer adam --beta1 0.9 --beta2 0.999 --lr 0.01 --clip --dataset citeseer --data-root /data/graphs`

This will start training the GATv2 model on the Citeseer dataset for 200 epochs. You should see output printed to the console, including the initial configuration summary and per-epoch metrics. For example, the program prints the configuration as:

    Configuration:
      Number of layers: 3
      Epochs: 200
      Attention heads: [4, 1, 1]
      Output dimensions: [64, 32, 16]
      Gradient clipping: true
      Optimizer: adam
      Learning rate: 0.01
    
    Using dataset: citeseer
    Dataset path: /data/graphs/citeseer/
    Max degree = ... 
    Number of classes = 6
    Graph loaded: 3327 nodes, 4732 edges, input_feature_vector_dim = 3703
    ...
After initialization, each epoch will be logged. You will see lines like:

    Epoch 1
    Avg Loss: 1.791234, Accuracy: 54.32%  total time: 6372.27 ms
    Epoch 2
    Avg Loss: 1.524678, Accuracy: 61.08%  total time: 6362.27 ms
    ...

The accuracy reported is on the training set (since this implementation does not separately evaluate on a test set in the code provided). Model will be extend to evaluate on validation/test splits as needed later.

 - **Note:** The CLI argument parsing in the code is order-sensitive in one case: it first parses `--num-layers` to know how many values to
   expect for heads and outdims. So you should specify `--num-layers`
   before `--heads` and `--outdims` in the argument list. In general,
   the examples above respect this order. If you forget to do this, the
   program might not parse the heads/outdims correctly.

## Repository Structure
After cloning, you should see the following key files and folders:

    ├── GATv2_node_based.cu        # Source code for node-centric implementation
    ├── GATv2_edge_based.cu        # Source code for edge-centric implementation
    ├── data/                      # (Optional) Data directory, if you put datasets here
    │   ├── cora/                  # Example dataset folder (Cora)
    │   │   ├── features.txt
    │   │   ├── row_ptr.txt
    │   │   ├── col_idx.txt
    │   │   └── labels.txt
    │   └── citeseer/              # Another dataset folder (Citeseer)
    │       └── ... (files similar to above)
    └── README.md                  # This readme file

### Additional Infomrmation:
The implementation uses random initialization for weights (Xavier/Glorot initialization using cuRAND). Each run may result in slightly different results due to randomness, especially when using multiple epochs and a non-deterministic order of floating-point operations (e.g., due to atomic adds). This is normal in deep learning training. You can set a fixed random seed in the code if reproducibility is needed (currently it seeds with `time(NULL)` for parameter init).

## Conclusion
By following this README, you should be able to compile and run the GATv2 node classification code. The provided implementations offer insight into how graph neural network computations can be carried out in parallel on GPUs. Whether you choose the node-centric or edge-centric approach, the model will leverage GATv2's dynamic attention mechanism to learn expressive node embeddings for classification tasks.
