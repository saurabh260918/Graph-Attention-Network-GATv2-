%%writefile cuda.cu
#include <cuda_runtime.h>
#include<cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <string> 
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#define MAX_OUT_DIM 17
#define MAX_IN_DIM 50




void load_features(const std::string& filename, float** h_features, int& num_nodes, int& input_dim) {
    std::ifstream file(filename);
    std::string line;
    std::vector<float> values;

    num_nodes = 0;
    input_dim = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float val;
        int dim_count = 0;
        while (iss >> val) {
            values.push_back(val);
            dim_count++;
        }
        if (input_dim == 0)
            input_dim = dim_count;
        else if (dim_count != input_dim) {
            std::cerr << "Inconsistent input_dim on line " << num_nodes << std::endl;
            exit(1);
        }
        num_nodes++;
    }

    *h_features = new float[values.size()];
    std::copy(values.begin(), values.end(), *h_features);
}

void load_int_array(const std::string& filename, int** arr, int& length) {
    std::ifstream file(filename);
    std::vector<int> values;
    int val;
    while (file >> val) {
        values.push_back(val);
    }
    length = values.size();
    *arr = new int[length];
    std::copy(values.begin(), values.end(), *arr);
}


//-------------------------------------------------------------------------------------//


// leaky relu operation_element-wise
__device__ float leaky_relu(float x, float alpha=0.01f) {
    return x > 0 ? x : alpha * x;
}

// Dot product of two vectors of length len
__device__ float dot(const float* a, const float* b, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) sum += a[i] * b[i];
    return sum;
}

// Matrix-vector multiplication: y = M * x
__device__ void matvec(const float* M, const float* x, float* y, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        y[i] = 0.0f;
        for (int j = 0; j < cols; ++j) y[i] += M[i * cols + j] * x[j];
    }
}

// Concatenate two vectors: [a, b] -> out
__device__ void concat(const float* a, const float* b, float* out, int len_a, int len_b) {
    for (int i = 0; i < len_a; ++i) out[i] = a[i];
    for (int i = 0; i < len_b; ++i) out[len_a + i] = b[i];
}

// Compute softmax over a small array (for a node's neighbors)
__device__ void softmax(float* scores, int len) {       // scores is a pointer to an array of attenstion scores of length len of a perticular node.
    float max_val = scores[0];
    for (int i = 1; i < len; ++i) if (scores[i] > max_val) max_val = scores[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }
    for (int i = 0; i < len; ++i) scores[i] /= sum;
}

__global__ void extract_W_head_right_kernel(
    const float* d_W,           // flattened [all layers]
    float* d_W_head_right,      // flattened [all layers]
    const int* head,            // [num_layers]
    const int* out_dim,         // [num_layers]
    const int* in_dim,          // [num_layers]
    int num_layers
) {
    int l = blockIdx.x;  // layer index
    int h = blockIdx.y;  // head index
    int o = threadIdx.x; // output dim row

    if (l < num_layers && h < head[l] && o < out_dim[l]) {
        // Compute offsets for this layer
        int w_offset = 0, w_right_offset = 0;
        for (int i = 0; i < l; ++i) {
            w_offset      += head[i] * out_dim[i] * 2 * in_dim[i];
            w_right_offset += head[i] * out_dim[i] * in_dim[i];
        }
        w_offset      += h * out_dim[l] * 2 * in_dim[l] + o * 2 * in_dim[l];
        w_right_offset += h * out_dim[l] * in_dim[l] + o * in_dim[l];

        // Copy right half of the row
        const float* src = d_W + w_offset + in_dim[l];
        float* dst = d_W_head_right + w_right_offset;
        for (int col = 0; col < in_dim[l]; ++col)
            dst[col] = src[col];
    }
}






__global__ void xavier_init_kernel(
    float* d_w,      // [num_layers][num_heads][out_dim][2*in_dim]
    float* d_a,      // [num_layers][num_heads][out_dim]
    float* d_wo,     // [C][out_dim_last_layer]
    const int*   head,     // [num_layers]  (each element is the number of heads per layer)
    const int*   in_dim,   // [num_layers]  (each element is the input dimension for that layer)
    const int*   out_dim,  // [num_layers]  (each element is the output dimension for that layer)
    int C,               // Number of output classes
    int    num_layers,
    unsigned int seed // Seed for random number generation
) {
    int l = blockIdx.x; // layer
    int h = blockIdx.y; // head
    int o = threadIdx.x; // output dimension index

    if (l < num_layers && h < head[l] && o < out_dim[l]) {
        // Xavier uniform limits
        int in_d = in_dim[l];
        int out_d = out_dim[l];
        float limit = sqrtf(6.0f / (2 * in_d + out_d));
        // Compute offset for d_w
        int w_offset = 0;
        for (int i = 0; i < l; ++i)
            w_offset += head[i] * out_dim[i] * 2 * in_dim[i];
        w_offset += h * out_dim[l] * 2 * in_dim[l];
        w_offset += o * 2 * in_dim[l];

        // Compute offset for d_a
        int a_offset = 0;
        for (int i = 0; i < l; ++i)
            a_offset += head[i] * out_dim[i];
        a_offset += h * out_dim[l];
        a_offset += o;


        // Initialize d_w: [out_dim][2*in_dim]
        for (int j = 0; j < 2 * in_d; ++j) {
            // Use a simple LCG for random number generation
            unsigned int local_seed = seed ^ (l * 73856093) ^ (h * 19349663) ^ (o * 83492791) ^ (j * 2654435761);
            float rnd = ((local_seed % 100000) / 100000.0f) * 2.0f * limit - limit;
            d_w[w_offset + j] = rnd;
        }
        // Initialize d_a: [out_dim]
        unsigned int local_seed = seed ^ (l * 1234567) ^ (h * 9876543) ^ (o * 1928374);
        float rnd = ((local_seed % 100000) / 100000.0f) * 2.0f * limit - limit;
        d_a[a_offset] = rnd;
    }

     // === Wo initialization ===
    if (l == num_layers - 1 && h == 0) {
        int out_d = out_dim[l];
        if (o >= out_d) return;
        float limit = sqrtf(6.0f / (C + out_d));
        // Each thread handles one output column of Wo
        for (int r = 0; r < C; ++r) {
            unsigned int local_seed = seed ^ (o * 99991) ^ (r * 88883) ^ 424242;
            float rnd = ((local_seed % 100000) / 100000.0f) * 2.0f * limit - limit;
            d_wo[r * out_d + o] = rnd;  // d_wo is [C x out_dim_last_layer] in row-major
        }
        
    }

    //for l==0, h==0 print the d_w in proper 2-D shape and d_a as a vector 
    // if (l == 0 && h == 0 && o == 0) {
    //     printf("d_w for layer %d, head %d:\n", l, h);
    //     for (int i = 0; i < out_dim[l]; ++i) {
    //         for (int j = 0; j < 2 * in_dim[l]; ++j) {
    //             printf("%f  ", d_w[i * 2 * in_dim[l] + j]);
    //         }
    //         printf("\n");
    //     }

    //     printf("d_a for layer %d, head %d:\n", l, h);
    //     for (int i = 0; i < out_dim[l]; ++i) {
    //         printf("%f ", d_a[i]);
    //     }
    //     printf("\n");
    // }

    // print d_wo for 0th class only as a sample to debug
    // if (l == num_layers - 1 && h == 0 && o == 0) {
    //     printf("d_wo for class 0:\n");
    //     for (int d = 0; d < out_dim[l]; ++d) {
    //         printf("%f ", d_wo[d]);
    //     }
    //     printf("\n");
    // }

    

}



__global__ void gatv2_forward_kernel(
    int N, int in_dim, int out_dim, int num_heads,
    const float*  d_features,    // [N][in_dim] it is the d_out of the previous layer.
    const int*  d_row_ptr,       // [N+1]
    const int*  d_col_idx,       // [num_edges]
    const float*  d_W,           // [num_heads][out_dim][2*in_dim]
    const float* d_W_head_right,             // [num_heads][out_dim][in_dim]
    const float*  d_a,           // [num_heads][out_dim]
    float* d_out,                            // [N][num_heads][out_dim] or [n][out_dim] if is_last_layer is true.
    bool is_last_layer, int max_degree
) {
    int node = blockIdx.x;        // Each thread block processes a node per head. total blocks (grid size) = N* num_heads. grid x-direction is node, y-direction is head.
    int head = blockIdx.y; 
           
    if (node >= N || head >= num_heads || threadIdx.x>=max_degree) return;

      //printf("check_kernel_start\n");

    // Shared memory for attention scores and softmax
    extern __shared__ float shared_mem[];
    float* attn_scores = shared_mem; // size = max_degree    //Assuming max degree is equal to the number of threads in the block 
    float* head_output = &shared_mem[max_degree]; // [out_dim]     //it will start after the attn_scores array in shared memory. Each thread will write its output to this array.
    int row_start = d_row_ptr[node];     
    int row_end = d_row_ptr[node + 1];
    int degree = row_end - row_start;     // Number of neighbors for the current node.

    // 1. Compute attention scores e_{ij}
    for (int nbr_idx = threadIdx.x; nbr_idx < degree; nbr_idx += blockDim.x) {
        int j = d_col_idx[row_start + nbr_idx];        // we are iterating over the neighbors of the Node 'node' in the graph. j is the neighbor node index.

        // Concatenate x_i and x_j
        float concat_x[2 * MAX_IN_DIM];        //concatenated feature vector of the node and its neighbor
        concat(&d_features[node * in_dim], &d_features[j * in_dim], concat_x, in_dim, in_dim);      // &d_features[j * in_dim] is the starting address of the features of the neighbor node j.

        // Linear transformation: s = W * [x_i ; x_j]
        float s[MAX_OUT_DIM];     // output vector after linear transformation
        const float* W_head = &d_W[head * out_dim * 2 * in_dim];    // W_head points to the weight matrix for the current head.
        matvec(W_head, concat_x, s, out_dim, 2 * in_dim);

        // LeakyReLU
        for (int k = 0; k < out_dim; ++k) s[k] = leaky_relu(s[k]);

        // Attention score: e_{ij} = a^T * s
        const float* a_head = &d_a[head * out_dim];   // a_head points to the attention vector for the current head.
        attn_scores[nbr_idx] = dot(a_head, s, out_dim);
    }
    __syncthreads();

    // 2. Softmax over neighbors (in-place on attn_scores)
    if (threadIdx.x == 0) softmax(attn_scores, degree);
    __syncthreads();

    // 3. Aggregate neighbor features using attention scores
    if (threadIdx.x < out_dim)
    head_output[threadIdx.x] = 0.0f;     //since we do atomicAdd, we need to initialize the output vector to zero before accumulating values.
    __syncthreads();

    const float* W_head_right = d_W_head_right + head * out_dim * in_dim;    // W_head_right points to the right part of the weight matrix FOR THE CURRENY HEAD.

    for (int nbr_idx = threadIdx.x; nbr_idx < degree; nbr_idx += blockDim.x) {
        int j = d_col_idx[row_start + nbr_idx];

        // Use the matvec utility for W_head_right * x_j
        float W_xj[MAX_OUT_DIM];                                                 // output vector after multiplying the right part of the weight matrix with the neighbor's features
        matvec(W_head_right, &d_features[j * in_dim], W_xj, out_dim, in_dim);

        // Accumulate weighted neighbor features
        for (int k = 0; k < out_dim; ++k)
            atomicAdd(&head_output[k], attn_scores[nbr_idx] * W_xj[k]);     // Multiply attention score with the transformed neighbor features and accumulate to the output vector (per node per head wise).
    }

    __syncthreads();

    // Write output
    if (is_last_layer) {
    // Each block computes head_output for (node, head)
    // Each thread block: (node, head)
    if (threadIdx.x == 0) {
        for (int k = 0; k < out_dim; ++k) {
            // Atomic add each head's output to the node's output slot
            atomicAdd(&d_out[node * out_dim + k], head_output[k] / num_heads);
        }
    }
    } else {
        // For intermediate layers: concatenate output per head
        if (threadIdx.x == 0) {
            for (int k = 0; k < out_dim; ++k)
                d_out[node * num_heads * out_dim + head * out_dim + k] = head_output[k];
        }
    }

    //print d_out one node for debugging
    // if (node == 0 && head == 0 && threadIdx.x == 0) {
    //     printf("d_out for node %d: ", node);
    //     for (int k = 0; k < out_dim; ++k) {
    //         printf("%f ", d_out[node * out_dim + k]);
    //     }
    //     printf("\n");
    // }

    //for each layer print d_layer_inputs for node 0 only. if l==L-1 then size is [num_nodes][out_dim[L-1]] else size is [num_nodes][head[l]*out_dim[l]]
    //print for node 0 only from d_out. this is device code cout will not work, so we will use printf.
    // if (node == 0 && head == 0 && threadIdx.x == 0) {

    // if (node == 0 && head == 0 && threadIdx.x == 0) {
    //     if (is_last_layer) {
    //         printf("Last Layer output for node 0: ");
    //         for (int k = 0; k < out_dim; ++k) {
    //             printf("%f ", d_out[0 * out_dim + k]);
    //         }
    //         printf("\n");
    //     } else {
    //         printf("Layer output for node 0 for in_dim %d: ", in_dim);
    //         for (int k = 0; k < num_heads * out_dim; ++k) {
    //             printf("%f ", d_out[0 * num_heads * out_dim + k]);
    //         }
    //         printf("\n");
    //     }
    // }
   
    
}

// Kernel definition for GATv2 output calculation
__global__ void gatv2_output_kernel(
    const float* d_wo,           // Weight matrix W_o [C x out_dim_last]
    const float* d_last_layer_output, // Input from the previous layer [num_nodes x out_dim_last]
    float* d_z,                  // Output after transformation [num_nodes x C]
    float* d_y,                  // Final output after softmax [num_nodes x C]
    int num_nodes,               // Number of nodes (N)
    int C,                       // Output feature channels per node
    int out_dim_last_layer       // Feature dimension of the last layer (out_dim[L-1])
) {
    // Thread/node index
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    // Pointer to this node's input feature (from previous layer)
    const float* node_output = &d_last_layer_output[node * out_dim_last_layer];

    // Pointer to this node's (node_id) output in d_z
    float* node_z = &d_z[node * C];

    // Compute z_i = W_o * input_i using provided matvec (output goes to node_z)
    matvec(d_wo, node_output, node_z, C, out_dim_last_layer);

    // Softmax over channels for this node: softmax(z_i), result in-place at node_z
    softmax(node_z, C);

    // Copy results into d_y
    float* node_y = &d_y[node * C];
    for (int i = 0; i < C; ++i) {
        node_y[i] = node_z[i];
    }

    //print d_y for node 0 only for debugging
    if (node == 0) {
        printf("d_y for node 0: ");
        float sum=0;
        for (int k = 0; k < C; ++k) {
            printf("%f ", d_y[k]);
            sum += d_y[k];
        }
        printf("\n");
        printf("Sum of probabilities for node 0: %f\n", sum);
    }
}



// __global__ void test_kernel() {
//     if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
//         printf("✅ Test kernel is working\n");
//     }
// }




int main() {
    // 1. Load graph in CSR format
    int num_nodes, num_edges, input_dim;
    float* h_features;    // [num_nodes][input_dim]
    int* h_row_ptr;       // [num_nodes+1]
    int* h_col_idx;       // [num_edges]

     // Load features
    load_features("/content/features.txt", &h_features, num_nodes, input_dim);

    // Load row_ptr
    int row_ptr_len;
    load_int_array("/content/row_ptr.txt", &h_row_ptr, row_ptr_len);
    if (row_ptr_len != num_nodes + 1) {
        std::cerr << "Invalid row_ptr length\n";
        return 1;
    }

    // Load col_idx
    int col_idx_len;
    load_int_array("/content/col_idx.txt", &h_col_idx, col_idx_len);
    num_edges = col_idx_len;

    // Now we can use h_features, h_row_ptr, h_col_idx

    // std::cout << "Loaded graph with " << num_nodes << " nodes, " << num_edges
    //           << " edges, input_dim: " << input_dim << std::endl;
    
    // //in below line print first 10 nodes h_row_ptr
    // for(int i = 0; i < 10; ++i) {
    //     std::cout << "h_row_ptr[" << i << "] = " << h_row_ptr[i] << " ";
    // }
    // std::cout << std::endl;
    // // Print first 10 col_idx
    // std::cout << "First 10 col_idx: ";
    // for (int i = 0; i < std::min(10, num_edges); ++i) {
    //     std::cout << h_col_idx[i] << " ";
    // }
    // std::cout << std::endl;

    // // Print first first node features.
    // std::cout << "First node features: ";
    // for (int i = 0; i < input_dim; ++i) {
    //     std::cout << h_features[i] << " ";
    // }
    // std::cout << std::endl;

   
    
    int max_degree = 20 ;      //hardcoded for now, but can be computed from h_row_ptr traversal.

    //std::cout << "Max degree = " << max_degree << std::endl;

     // 2. Define GATv2 architecture
    const int L = 3; // Example: 3 layers
    int head[L] = {2, 2, 2};         // Number of heads per layer
    int out_dim[L] = {10, 10, 10};    // Output dim per head per layer
    int in_dim[L]= {input_dim}; // Input dim for first layer, subsequent layers will be computed based on previous layer's output.
    in_dim[0] = input_dim;
    for (int l = 1; l < L; ++l)
        in_dim[l] = head[l-1] * out_dim[l-1];
    int C = 20; // C class classification problem.

    // 3. Declare device pointers for all parameters and caches
    int* d_head;          // Device array for number of heads per layer
    int* d_out_dim;      // Device array for output dimensions per layer
    int* d_in_dim;      // Device array for input dimensions per layer
    float* d_w;         // flat Weight matrices array for all layer
    float* d_a;         // flat Attention vectors array for all layers
    float* d_w_right;   // flat Right-part of weights matrix for all layer
    float* d_features;     // Device input features
    int* d_row_ptr;    // Device CSR row pointer
    int* d_col_idx;    // Device CSR edge array
    float* d_layer_outputs[L]; // Output buffers per layer
    float* d_wo;       // Device linear transformation weight matrix of size C X out_dim.
    float* d_z;        // output after linear transformation. size is number of nodes X C.
    float* d_y;        // output probabilities [N][C]

     // Start timer
    auto start = std::chrono::high_resolution_clock::now();

     // 4. Initialize parameters and allocate memory
    cudaError_t err;
    // 1. Malloc and memcpy for graph data
    err = cudaMalloc(&d_features, num_nodes * in_dim[0] * sizeof(float));
    //printf("cudaMalloc d_features: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_features, h_features, num_nodes * in_dim[0] * sizeof(float), cudaMemcpyHostToDevice);
    //printf("cudaMemcpy d_features: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&d_row_ptr, (num_nodes+1) * sizeof(int));
    //printf("cudaMalloc d_row_ptr: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_row_ptr, h_row_ptr, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice);
    //printf("cudaMemcpy d_row_ptr: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&d_col_idx, num_edges * sizeof(int));
    //printf("cudaMalloc d_col_idx: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_col_idx, h_col_idx, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    //printf("cudaMemcpy d_col_idx: %s\n", cudaGetErrorString(err));

    

    for (int l = 0; l < L; ++l) {
        size_t output_size;
        if (l == L - 1) {
            // Last layer: average heads, only out_dim[l] per node
            output_size = num_nodes * out_dim[l];
        } else {
            // Non-last layer: concatenate heads
            output_size = num_nodes * head[l] * out_dim[l];
        }
        cudaMalloc(&d_layer_outputs[l], output_size * sizeof(float));
        //printf("cudaMalloc d_layer_outputs[%d]: %s\n", l, cudaGetErrorString(err));
    }

    
    err = cudaMalloc(&d_head, L * sizeof(int));
    //printf("cudaMalloc d_head: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_head, head, L * sizeof(int), cudaMemcpyHostToDevice);
    //printf("cudaMemcpy d_head: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&d_in_dim, L * sizeof(int));
    //printf("cudaMalloc d_in_dim: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_in_dim, in_dim, L * sizeof(int), cudaMemcpyHostToDevice);
    //printf("cudaMemcpy d_in_dim: %s\n", cudaGetErrorString(err));

    err = cudaMalloc(&d_out_dim, L * sizeof(int));
    //printf("cudaMalloc d_out_dim: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_out_dim, out_dim, L * sizeof(int), cudaMemcpyHostToDevice);
    //printf("cudaMemcpy d_out_dim: %s\n", cudaGetErrorString(err));

    // Compute total size needed for all layers
    size_t total_w = 0, total_a = 0;
    int w_offset[L]= {0}; // Offsets for weights
    int a_offset[L]= {0}; // Offsets for attention vectors
    for (int l = 0; l < L; ++l) {
        w_offset[l] = head[l] * out_dim[l] * 2 * in_dim[l];
        a_offset[l] = head[l] * out_dim[l];
        total_w += w_offset[l];
        total_a += a_offset[l];
    }
    cudaMalloc(&d_w, total_w * sizeof(float));
    cudaMalloc(&d_a, total_a * sizeof(float));
    cudaMalloc(&d_wo, C * out_dim[L - 1] * sizeof(float));
    cudaMalloc(&d_z, num_nodes * C * sizeof(float));
    cudaMalloc(&d_y, num_nodes * C * sizeof(float));



    // Compute total size needed for all layers' right-half weights
    size_t total_w_right = 0;
    int w_right_offset[L] = {0}; // Offsets for right-half weights
    for (int l = 0; l < L; ++l){
        w_right_offset[l] = head[l] * out_dim[l] * in_dim[l];
        total_w_right += w_right_offset[l];
    }

    cudaMalloc(&d_w_right, total_w_right * sizeof(float));


    int max_heads_per_layer = *std::max_element(head, head + L);  // Maximum number of heads across all layers
    int max_out_dim = *std::max_element(out_dim, out_dim + L); // Maximum output dimension across all layers


    // 3. Initialize weights and attention vectors using Xavier initialization
    dim3 grid(L, max_heads_per_layer);
    int block = max_out_dim;    // Maximum output dimension across all layers
    xavier_init_kernel<<<grid, block>>>(d_w, d_a, d_wo, d_head, d_in_dim, d_out_dim, C, L, time(NULL));
    cudaDeviceSynchronize();


    
    // This kernel will extract the right half of the weights for each head and store them in d_w_right.
    extract_W_head_right_kernel<<<grid, block>>>(  d_w, d_w_right, d_head, d_out_dim, d_in_dim, L );
    cudaDeviceSynchronize();

  
    // 5. Forward pass for each layer
    float* d_layer_inputs = d_features;
    for (int l = 0; l < L; ++l) {
        dim3 grid(num_nodes, head[l]);
        int block = max_degree; // or set to max degree or a tuned value
        size_t shared_mem = (max_degree+out_dim[l]) * sizeof(float); 
        bool is_last_layer = (l == L - 1);
        const float* d_w_l = d_w + (l > 0 ? w_offset[l-1] : 0); // Left half of weights for this layer
        const float* d_a_l = d_a + (l > 0 ? a_offset[l-1] : 0); // Attention vector for this layer
        const float* d_w_right_l = d_w_right + (l > 0 ? w_right_offset[l-1] : 0); // Right half of weights for this layer

        gatv2_forward_kernel<<<grid, block, shared_mem>>>( num_nodes, in_dim[l], out_dim[l], head[l], d_layer_inputs, d_row_ptr, d_col_idx, d_w_l, d_w_right_l, d_a_l, d_layer_outputs[l], is_last_layer, max_degree);
        //printf("Kernel launched for layer %d with grid (%d, %d) and block %d\n", l, num_nodes, head[l], block);
        
        //Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error after kernel launch at layer %d: %s\n",l,  cudaGetErrorString(err));
              
        }
        cudaDeviceSynchronize();

        // Next layer input is output from this layer
        d_layer_inputs = d_layer_outputs[l];
      
    }
    int threads_per_block = 128;
    int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    gatv2_output_kernel<<<num_blocks, threads_per_block>>>(d_wo,  d_layer_inputs, d_z, d_y, num_nodes, C, out_dim[L - 1]);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after gatv2_output_kernel launch: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();
    // 6. Copy output back to host
    float* h_y = new float[num_nodes * C];
    err = cudaMemcpy(h_y, d_y, num_nodes * C * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying output to host: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    


    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    std::cout <<  " total time: " << elapsed.count() << " ms" << std::endl;


}

