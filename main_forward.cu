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
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define MAX_OUT_DIM 1000
#define MAX_IN_DIM 1500




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




int compute_max_degree(const int* h_row_ptr, int num_nodes) {
    int max_degree = 0;
    //int num_nodes = h_row_ptr.size() - 1;
    for (int i = 0; i < num_nodes; ++i) {
        int degree = h_row_ptr[i + 1] - h_row_ptr[i];
        if (degree > max_degree) {
            max_degree = degree;
        }
    }
    return max_degree;
}


float compute_loss_and_accuracy(int N, float* d_loss, int* d_correct) {
    thrust::device_ptr<float> dev_losses(d_loss);
    thrust::device_ptr<int> dev_corrects(d_correct);
    float total_loss = thrust::reduce(dev_losses, dev_losses + N, 0.0f, thrust::plus<float>());
    int total_correct = thrust::reduce(dev_corrects, dev_corrects + N, 0, thrust::plus<int>());
    float avg_loss = total_loss / N;
    float accuracy = static_cast<float>(total_correct) / N;
    printf("Avg Loss: %f, Accuracy: %.2f%%\n", avg_loss, 100.0f * accuracy);
    return avg_loss;
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
 

}



__global__ void gatv2_forward_kernel(
    int N, int in_dim, int out_dim, int num_heads,
    const float* d_features,      // [N][in_dim]
    const int* d_row_ptr,         // [N+1]
    const int* d_col_idx,         // [num_edges]
    const float* d_W,             // [num_heads][out_dim][2*in_dim]
    const float* d_a,             // [num_heads][out_dim]
    float* d_h,                  //  [N][num_heads][out_dim] or [N][out_dim]      //pre-nonlinearity output of layers
    float* d_out,                 // [N][num_heads][out_dim] or [N][out_dim] if last layer
    float* attn_score,            // [N][num_heads][max_degree]
    float* attn_coeff,            // [N][num_heads][max_degree]
    float* d_leakyrelu,           // [N][num_heads][max_degree][out_dim]
    float* d_s,                   // [N][num_heads][max_degree][out_dim]
    bool is_last_layer,
    int max_degree
) {
    int node = blockIdx.x;
    int head = blockIdx.y;

    if (node >= N || head >= num_heads || threadIdx.x >= max_degree) return;

    extern __shared__ float shared_mem[];
    float* attn_scores = shared_mem;         // [max_degree]
    float* head_output = &shared_mem[max_degree]; // [out_dim]
    int row_start = d_row_ptr[node];
    int row_end = d_row_ptr[node + 1];
    int degree = row_end - row_start;

    if (threadIdx.x >= degree) return;

    int j = d_col_idx[row_start + threadIdx.x];

    // Concatenate x_i and x_j
    float concat_x[2 * MAX_IN_DIM];
    concat(&d_features[node * in_dim], &d_features[j * in_dim], concat_x, in_dim, in_dim);

    float s[MAX_OUT_DIM];
    const float* W_head = &d_W[head * out_dim * 2 * in_dim];
    matvec(W_head, concat_x, s, out_dim, 2 * in_dim);

    // Store pre-activation s_ij for this node/head/neigh
    int base_s = (((node * num_heads + head) * max_degree) + threadIdx.x) * out_dim;
    for(int k = 0; k < out_dim; ++k) {
        d_s[base_s + k] = s[k];  // Save s_ij before nonlinearity
    }

    // LeakyReLU & store per-edge per-dim
    for (int k = 0; k < out_dim; ++k) s[k] = leaky_relu(s[k]);
    for(int k = 0; k < out_dim; ++k) {
        d_leakyrelu[base_s + k] = s[k];  // Store leaky_relu(s_ij) per neighbor/edge
    }

    // Compute attention score: e_ij = a^T s (after nonlinearity)
    const float* a_head = &d_a[head * out_dim];
    float attnscore_val = dot(a_head, s, out_dim);
    attn_scores[threadIdx.x] = attnscore_val;
    attn_score[(node * num_heads + head) * max_degree + threadIdx.x] = attnscore_val;
    __syncthreads();

    if (threadIdx.x == 0) softmax(attn_scores, degree);
    //store softmax(attn_scores) in attn_coeff by each thread since total active threads < max_degree
    attn_coeff[(node * num_heads + head) * max_degree + threadIdx.x] = attn_scores[threadIdx.x];
    __syncthreads();

    for (int i = threadIdx.x; i < out_dim; i += blockDim.x)
        head_output[i] = 0.0f;
    __syncthreads();

    // Apply right-half weight to x_j
    // const float* W_head_right = d_W_head_right + head * out_dim * in_dim;
    // float W_xj[MAX_OUT_DIM];
    // matvec(W_head_right, &d_features[j * in_dim], W_xj, out_dim, in_dim);

    float W_xj[MAX_OUT_DIM];
    int base = head * out_dim * 2 * in_dim;
    for (int od = 0; od < out_dim; ++od) {
        float sum = 0.0f;
        for (int id = 0; id < in_dim; ++id) {
            float w_val = d_W[base + od * 2 * in_dim + (in_dim + id)];
            float x_val = d_features[j * in_dim + id];
            sum += w_val * x_val;
        }
        W_xj[od] = sum;
    }

    // Accumulate weighted neighbor features   h_i
    for (int k = 0; k < out_dim; ++k)
        atomicAdd(&head_output[k], attn_scores[threadIdx.x] * W_xj[k]);

    __syncthreads();

    if (!is_last_layer) {
        // Nonlinearity after aggregation (per head)
        for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
            d_h[node * num_heads * out_dim + head * out_dim + i] = head_output[i];
            head_output[i] = leaky_relu(head_output[i]);
            d_out[node * num_heads * out_dim + head * out_dim + i] = head_output[i];
        }
    } else {
        // Final layer: average over heads
        if (threadIdx.x == 0) {
            for (int k = 0; k < out_dim; ++k)
                atomicAdd(&d_out[node * out_dim + k], head_output[k] / num_heads); // Aggregated output before nonlinearity
        }
        __syncthreads();
        // store PRE-nonlinearity (averaged-over-heads)
        if (threadIdx.x == 0 && head == 0) {
            for (int i = 0; i < out_dim; ++i) d_h[node * out_dim + i] = d_out[node * out_dim + i];
        }
        __syncthreads();
        // Nonlinearity after averaging
        if (head == 0) {
            for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
                float pre_nlin = d_out[node * out_dim + i];
                float post_nlin = leaky_relu(pre_nlin);
                d_out[node * out_dim + i] = post_nlin;
            }
        }
    }
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

__global__ void compute_output_gradients(
    const float* d_y,        // [N][C], softmax output
    const int* d_labels,     // [N],   true labels
    const float* d_hL,       // [N][out_dim_L], PRE-nonlinearity (input to activation)
    const float* d_HL,       // [N][out_dim_L], POST-nonlinearity (output of activation)
    const float* d_wo,       // [C][out_dim_L], output linear W
    float* grad_d_wo,        // [C][out_dim_L], output: grad for W_o
    float* grad_d_hL,        // [N][num_heads][out_dim_L], output: grad for h_i^L (pre-activation) per head
    int N, int C, int out_dim_L,
    int num_heads,  
    float negative_slope     // LeakyReLU slope
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    // Step 1: compute dL/dz = y_hat - y
    extern __shared__ float shared_memory[];
    float* dL_dz = &shared_memory[0];

    for (int c = 0; c < C; ++c) {
        dL_dz[c] = d_y[node*C + c] - (c == d_labels[node] ? 1.0f : 0.0f);
    }

    // Step 2: accumulate grad for W_o: dL/dWo += dL/dz * H_i_L^T
    for (int c = 0; c < C; ++c) {
        for (int d = 0; d < out_dim_L; ++d) {
            atomicAdd(&grad_d_wo[c*out_dim_L + d], dL_dz[c] * d_HL[node*out_dim_L + d]);
        }
    }

    // Step 3: backprop to H^(L): dL/dH^(L) = W_o^T * dL/dz, we are storing this in sum variable //// store the pre-activation gradient in grad_d_hL for that node. it will be 1/num_heads of the total gradient
    float inv_heads = 1.0f / (float)num_heads;
    for (int d = 0; d < out_dim_L; ++d) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += d_wo[c*out_dim_L + d] * dL_dz[c];
        }
        //grad_d_HL[node*out_dim_L + d] = sum;
        //grad_hL= (grad_d_HL * derivative)/num_heads. //this is per head grad


        // Step 4: Chain rule through the non-linearity (LeakyReLU)
        float h_val = d_hL[node*out_dim_L + d];
        float derivative = (h_val > 0.0f) ? 1.0f : negative_slope;
        for (int h = 0; h < num_heads; ++h) {
            grad_d_hL[node * num_heads * out_dim_L + h * out_dim_L + d] =
                sum * derivative * inv_heads;   
        }
    }
}


//.........................................................................................................//

__global__ void gatv2_layer_backward(
    int N, int in_dim, int out_dim, int num_heads,
    const int* d_row_ptr, const int* d_col_idx,
    const float* d_x,         // [N][in_dim]
    const float* d_higher,    // [N][num_heads][out_dim]   //gradient of head outputs after activation //input from  above layer
    float* d_higher_pre, // [N][num_heads][out_dim]   //gradient of head outputs before activation
    float* d_h_pre,     // [N][num_heads][out_dim]   //head outputs before activation
    float* attn_coeff,  // [N][num_heads][max_degree]
    float* attn_score,  // [N][num_heads][max_degree]
    float* d_leakyrelu, // [N][num_heads][max_degree][out_dim]
    float* d_w,         // [num_heads][out_dim][2*in_dim]
    float* d_a,         // [num_heads][out_dim]
    float* d_s,         // [N][num_heads][max_degree][out_dim]
    float* grad_w,            // [num_heads][out_dim][2*in_dim]
    float* grad_a,            // [num_heads][out_dim]
    float* grad_x_lower,      // [N][in_dim]    // gradient of input features
    float negative_slope,
    int max_degree,
    bool last_layer // 0 if not last layer, 1 if last layer
) {
    int i = blockIdx.x;    // Node index
    int h = blockIdx.y;    // Head index
    int tid = threadIdx.x; // Neighbor offset
    //create a shared memory equal to max_degree+out_dim
    extern __shared__ float shared_memory[];    //size max_degree
    float* dL_d_h = &shared_memory[max_degree]; // size out_dim

    if(!last_layer){
        for (int od = tid; od < out_dim; od += blockDim.x) {
            float dL_d_H = d_higher[(i * num_heads + h) * out_dim + od];
            float h_val = d_h_pre[(i * num_heads + h) * out_dim + od];
            float deriv = (h_val > 0.0f) ? 1.0f : negative_slope;
            dL_d_h[od] = dL_d_H * deriv;
            d_higher_pre[(i * num_heads + h) * out_dim + od] = dL_d_h[od]; // Store pre-activation gradient in global memory
        }
    }
    
    else{
        for (int od = tid; od < out_dim; od += blockDim.x){
            dL_d_h[od] = d_higher[(i * num_heads + h) * out_dim + od];
            d_higher_pre[(i * num_heads + h) * out_dim + od] = dL_d_h[od];
        }

    }
    __syncthreads();


    int row_start = d_row_ptr[i];
    int row_end = d_row_ptr[i + 1];
    int deg = row_end - row_start;
    if (tid >= deg) return;

    int jj = row_start + tid;
    int j = d_col_idx[jj];

    // --- Step D.2: dL/d alpha_ij ---
    float dL_d_alpha_ij = 0.0f;
    for (int od = 0; od < out_dim; ++od) {
        //float dL_d_h = d_higher[(i * num_heads + h) * out_dim + od];
        for (int id = 0; id < in_dim; ++id) {
            float w_ = d_w[(h*out_dim*2*in_dim) + (od*2*in_dim) + (in_dim + id)];
            dL_d_alpha_ij += dL_d_h[od] * w_ * d_x[j*in_dim + id];
        }
    }
    // Store dL/d alpha_ij in shared memory for later use
    shared_memory[tid] = dL_d_alpha_ij;
    __syncthreads(); // Ensure all threads have written their values


    // --- Step D.4: grad_W direct ---
    float alpha = attn_coeff[(i * num_heads + h) * max_degree + (tid)];
    for (int od = 0; od < out_dim; ++od) {
        //float dL_d_h = d_higher[(i * num_heads + h) * out_dim + od];
        for (int id = 0; id < in_dim; ++id) {
            float x_j = d_x[j*in_dim + id];
            atomicAdd(&grad_w[(h*out_dim*2*in_dim) + (od*2*in_dim) + (in_dim + id)], alpha * dL_d_h[od] * x_j);    //here massive sequential addition possible
        }
    }

    // --- Step E.1: dL/d e_ij ---  
    float dL_d_e_ij = 0.0f;
    float alpha_ij = attn_coeff[(i * num_heads + h) * max_degree + tid];
    for (int kk = 0; kk < deg; ++kk) {
        int k = d_col_idx[row_start + kk];
        float alpha_ik = attn_coeff[(i * num_heads + h) * max_degree + kk];
        // For dL_d_alpha_ik, may require shared or global memory,
        // or recompute in another pass for full parallel safety
        float dL_d_alpha_ik = shared_memory[kk];
        dL_d_e_ij += dL_d_alpha_ik * alpha_ik * ((j == k ? 1.0f : 0.0f) - alpha_ij);
    }

    // --- Step E.2: grad_a ---
    int leaky_base = (((i * num_heads + h) * max_degree) + tid) * out_dim;
    for (int od = 0; od < out_dim; ++od) {
        float leaky_val = d_leakyrelu[leaky_base + od];
        float a_contrib = dL_d_e_ij * leaky_val;
        atomicAdd(&grad_a[h * out_dim + od], a_contrib);    // it can make slow, massive sequential addition
    }

    // --- Step E.3: grad_W via attention ---
    float* s_ij = &d_s[leaky_base]; // each [out_dim]
    float leaky_grad_val;
    for (int od = 0; od < out_dim; ++od) {
        leaky_grad_val = (s_ij[od] > 0) ? 1.0f : negative_slope;
        float elem = d_a[h * out_dim + od] * leaky_grad_val * dL_d_e_ij;
        for (int id = 0; id < 2 * in_dim; ++id) {
            float x_concat = (id < in_dim) ? d_x[i*in_dim + id] : d_x[j*in_dim + (id-in_dim)];
            float grad_contrib = elem * x_concat;
            atomicAdd(&grad_w[h * out_dim * 2 * in_dim + od * 2 * in_dim + id], grad_contrib);
        }
    }

    // ---- DIRECT GRADIENT FOR x_i ----
    // For node i as a neighbor of node j, accumulate to grad_x_lower[i][*] as per direct formula
    int offset = -1;
    for (int t = d_row_ptr[j]; t < d_row_ptr[j+1]; ++t) {
        if (d_col_idx[t] == i) {
            offset = t - d_row_ptr[j];
            break;
        }
    }


    float alpha_j_i = attn_coeff[(j * num_heads + h) * max_degree + offset];
    for (int od = 0; od < out_dim; ++od) {
        float dL_d_hj = d_higher_pre[(j * num_heads + h) * out_dim + od];
        // Right-part of W (W_right: [out_dim][in_dim]), maps neighbor features
        for (int id = 0; id < in_dim; ++id) {
            float W_right = d_w[(h * out_dim * 2 * in_dim) + (od * 2 * in_dim) + (in_dim + id)];
            // atomic add: node i, feature id
            atomicAdd(&grad_x_lower[i * in_dim + id],
                    alpha_j_i * W_right * dL_d_hj);
        }
    }

    // ---- INDIRECT GRADIENT FOR x_i ----
    for (int od = 0; od < out_dim; ++od) {
        // Compute LeakyReLU' for this dimension
        float leaky_grad = (s_ij[od] > 0) ? 1.0f : negative_slope;
        // Compute elementwise product with a
        float elem = d_a[h * out_dim + od] * leaky_grad * dL_d_e_ij;

        // Left part of W for this head/output: [out_dim][in_dim]
        for (int id = 0; id < in_dim; ++id) {
            float W_left = d_w[(h * out_dim * 2 * in_dim) + (od * 2 * in_dim) + id];
            float grad_contrib = elem * W_left;
            atomicAdd(&grad_x_lower[i * in_dim + id], grad_contrib);
        }
    }

}


__global__ void sgd_update_kernel(float* params, const float* grads, float lr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        params[i] -= lr * grads[i];
}


// Kernel for per-node loss and correct prediction
__global__ void compute_loss_accuracy_kernel(
    const float* d_y,         // [N][C] softmax, row-major
    const int* d_labels,      // [N], true class
    float* d_losses,          // [N], output: cross-entropy loss
    int* d_corrects,          // [N], output: 1/0 for accuracy
    int N, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int label = d_labels[idx];
    float prob = fmaxf(d_y[idx * C + label], 1e-10f); // avoid log(0)
    d_losses[idx] = -logf(prob);

    // Find predicted class (argmax)
    float maxval = d_y[idx * C];
    int pred = 0;
    for (int c = 1; c < C; ++c) {
        float val = d_y[idx * C + c];
        if (val > maxval) { maxval = val; pred = c; }
    }
    d_corrects[idx] = (pred == label);
}




int main() {
    // 1. Load graph in CSR format
    int num_nodes, num_edges, input_dim;
    float* h_features;    // [num_nodes][input_dim]
    int* h_row_ptr;       // [num_nodes+1]
    int* h_col_idx;       // [num_edges]
    int* h_labels;       // [num_nodes]

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

    //load labels
    int labels_len;
    load_int_array("/content/labels.txt", &h_labels, labels_len);
    if (labels_len != num_nodes) {
        std::cerr << "Invalid labels length\n";
        return 1;
    }

    int max_degree = compute_max_degree(h_row_ptr, num_nodes);
    std::cout << "Max degree = " << max_degree << std::endl;

    // auto itr = thrust::max_element(thrust::host, h_labels, h_labels + num_nodes);
    // int C = *itr + 1; // Number of classes, assuming labels are 0-indexed

     // 2. Define GATv2 architecture
    const int L = 3; // Example: 3 layers
    int C = 7; // C class classification problem.  //hardcoded for now.
    int head[L] = {1, 1, 1};         // Number of heads per layer
    int out_dim[L] = {1000, 500, 100};    // Output dim per head per layer
    // int out_dim[L];
    // int prev_dim = input_dim;
    // for (int i = 0; i < L; ++i) {
    //     out_dim[i] = prev_dim / 2;
    //     prev_dim = out_dim[i];
    // }
    int in_dim[L]= {input_dim}; // Input dim for first layer, subsequent layers will be computed based on previous layer's output.
    in_dim[0] = input_dim;
    for (int l = 1; l < L; ++l)
        in_dim[l] = head[l-1] * out_dim[l-1];
    

    // 3. Declare device pointers for all parameters and caches
    int* d_head;          // Device array for number of heads per layer
    int* d_out_dim;      // Device array for output dimensions per layer
    int* d_in_dim;      // Device array for input dimensions per layer
    float* d_w;         // flat Weight matrices array for all layer
    float* d_a;         // flat Attention vectors array for all layers
    float* d_features;     // Device input features
    int* d_row_ptr;    // Device CSR row pointer
    int* d_col_idx;    // Device CSR edge array
    int* d_labels;     // Device labels array
    float* d_layer_outputs[L]; // Output buffers per layer
    float* d_h[L];       // pre-nonlinearity output of hidden layers
    float* d_wo;       // Device linear transformation weight matrix of size C X out_dim.
    float* d_z;        // output after linear transformation. size is number of nodes X C.
    float* d_y;        // output probabilities [N][C]
    //float* d_loss;     // [N],   loss per node
    float* attn_score[L]; // Attention scores per layer [N][num_heads][max_degree]
    float* attn_coeff[L]; // Attention coefficients per layer [N][num_heads][max_degree]
    float* d_leakyrelu[L]; // LeakyReLU outputs per layer
    float* d_s[L];        // pre-nonlinearity output of each edge per layer
    float* grad_wo;   // Gradient of output layer weights [C][out_dim_last_layer]
    float* grad_d_w; // Gradient of weights for all layers [num_layers][num_heads][out_dim][2*in_dim]
    float* grad_d_a; // Gradient of attention vectors for all layers [num_layers][num_heads][out_dim]
    float negative_slope = 0.01f; // LeakyReLU slope

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
    err= cudaMalloc(&d_labels, num_nodes * sizeof(int));
    //printf("cudaMalloc d_labels: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_labels, h_labels, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    //printf("cudaMemcpy d_labels: %s\n", cudaGetErrorString(err));

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
        cudaMalloc(&d_h[l], output_size * sizeof(float));
        //printf("cudaMalloc d_layer_outputs[%d]: %s\n", l, cudaGetErrorString(err));
    }
    for (int l = 0; l < L; ++l) {
        cudaMalloc(&attn_score[l], num_nodes * head[l] * max_degree * sizeof(float));
        cudaMalloc(&attn_coeff[l], num_nodes * head[l] * max_degree * sizeof(float));
        cudaMalloc(&d_leakyrelu[l], num_nodes * head[l] * max_degree * out_dim[l] * sizeof(float));
        cudaMalloc(&d_s[l], num_nodes * head[l] * max_degree * out_dim[l] * sizeof(float));
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
    cudaMalloc(&grad_wo, C * out_dim[L-1] * sizeof(float));
    cudaMalloc(&grad_d_w, total_w * sizeof(float));
    cudaMalloc(&grad_d_a, total_a * sizeof(float));
    

    int max_heads_per_layer = *std::max_element(head, head + L);  // Maximum number of heads across all layers
    int max_out_dim = *std::max_element(out_dim, out_dim + L); // Maximum output dimension across all layers


    // 3. Initialize weights and attention vectors using Xavier initialization
    dim3 grid(L, max_heads_per_layer);
    int block = max_out_dim;    // Maximum output dimension across all layers
    xavier_init_kernel<<<grid, block>>>(d_w, d_a, d_wo, d_head, d_in_dim, d_out_dim, C, L, time(NULL));
    cudaDeviceSynchronize();

    float* d_loss;      // [num_nodes]
    int*   d_correct;   // [num_nodes]
    cudaMalloc(&d_loss,    num_nodes * sizeof(float));
    cudaMalloc(&d_correct, num_nodes * sizeof(int));

    // Allocate output gradient buffers: one per layer, size: N x in_dim[l]
    float** output_gradients = new float*[L];
    for (int l = 0; l < L; ++l) {
        size_t size = (size_t)num_nodes * in_dim[l] * sizeof(float);
        cudaMalloc(&output_gradients[l], size);
        cudaMemset(output_gradients[l], 0, size);  // Initialize to zero
    }
    float* input_gradient_last_layer; // Gradient buffer for the last layer input
    float* grad_d_higher_pre; // Gradient of pre-activation output of higher layer
    cudaMalloc(&grad_d_higher_pre, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float));
    cudaMemset(grad_d_higher_pre, 0, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float)); // Initialize to zero
    // Allocate input gradient buffer for the last layer: size: N x (last layer no. of heads x out_dim_last layer)
    cudaMalloc(&input_gradient_last_layer, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float));
    cudaMemset(input_gradient_last_layer, 0, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float)); // Initialize to zero

    for (int epoch = 0; epoch < 4; ++epoch) {

        // b. Forward pass for each layer
        float* d_layer_inputs = d_features;
        for (int l = 0; l < L; ++l) {
            dim3 grid(num_nodes, head[l]);
            int block = max_degree;
            size_t shared_mem = (max_degree + out_dim[l]) * sizeof(float);
            bool is_last_layer = (l == L - 1);
            const float* d_w_l = d_w + (l > 0 ? w_offset[l-1] : 0);
            const float* d_a_l = d_a + (l > 0 ? a_offset[l-1] : 0);

            gatv2_forward_kernel<<<grid, block, shared_mem>>>(
                num_nodes, in_dim[l], out_dim[l], head[l], d_layer_inputs,
                d_row_ptr, d_col_idx, d_w_l, d_a_l, d_h[l], d_layer_outputs[l],
                attn_score[l], attn_coeff[l], d_leakyrelu[l], d_s[l], is_last_layer, max_degree
            );
            cudaDeviceSynchronize();
            d_layer_inputs = d_layer_outputs[l];
        }

        int threads_per_block = 128;
        int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

        gatv2_output_kernel<<<num_blocks, threads_per_block>>>(d_wo,  d_layer_inputs, d_z, d_y, num_nodes, C, out_dim[L - 1]);

        // c. Calculate loss and accuracy ----

        int threads = 256;
        int blocks = (num_nodes + threads - 1) / threads;
        compute_loss_accuracy_kernel<<<blocks, threads>>>(
            d_y, d_labels, d_loss, d_correct, num_nodes, C);
        cudaDeviceSynchronize();

        compute_loss_and_accuracy(num_nodes, d_loss, d_correct);

        float* grad_input = input_gradient_last_layer;
        // d. Backward pass
        // 1: Output layer gradient
        size_t shared_memory = (C) * sizeof(float);
        compute_output_gradients<<<num_blocks, threads_per_block, shared_memory>>>(
            d_y, d_labels, d_h[L-1], d_layer_outputs[L-1], d_wo, grad_wo,
            grad_input, num_nodes, C, out_dim[L-1], head[L-1], negative_slope
        );
        cudaDeviceSynchronize();

        // 2: Loop backward through GAT layers
        for (int l = L-1; l >= 0; --l) {
            dim3 grid(num_nodes, head[l]);
            int block = max_degree;
            size_t shared_mem = (max_degree + out_dim[l]) * sizeof(float);
            float* d_w_l = d_w + (l > 0 ? w_offset[l-1] : 0);
            float* d_a_l = d_a + (l > 0 ? a_offset[l-1] : 0);
            bool is_last_layer = (l == L - 1);
            float* grad_output = output_gradients[l];
            gatv2_layer_backward<<<grid, block, shared_mem>>>(
                num_nodes, in_dim[l], out_dim[l], head[l],
                d_row_ptr, d_col_idx,
                d_layer_outputs[(l > 0) ? l-1 : 0], // d_x
                grad_input, grad_d_higher_pre, d_h[l], attn_coeff[l], attn_score[l],
                d_leakyrelu[l], d_w_l, d_a_l,
                grad_d_w + ((l > 0) ? w_offset[l-1] : 0),
                grad_d_a + ((l > 0) ? a_offset[l-1] : 0), d_s[l],
                grad_output, negative_slope, max_degree, is_last_layer
            );
            cudaDeviceSynchronize();

            // Prepare input gradient for next iteration down the stack
            grad_input = grad_output; // input gradient of this layer becomes output gradient of next lower layer
            
        }

        // -----> *** 5. PARAMETER UPDATE SECTION —  ***
        float lr = 0.01f; // Set your learning rate

        int block_size = 256;

        // --- Update d_w ---
        int num_blocks_w = (total_w + block_size - 1) / block_size;
        sgd_update_kernel<<<num_blocks_w, block_size>>>(d_w, grad_d_w, lr, total_w);

        // --- Update attention vectors d_a ---
        int num_blocks_a = (total_a + block_size - 1) / block_size;
        sgd_update_kernel<<<num_blocks_a, block_size>>>(d_a, grad_d_a, lr, total_a);

        // --- Update output weights d_wo ---
        int total_wo = C * out_dim[L - 1];
        int num_blocks_wo = (total_wo + block_size - 1) / block_size;
        sgd_update_kernel<<<num_blocks_wo, block_size>>>(d_wo, grad_wo, lr, total_wo);

        cudaDeviceSynchronize(); 

        // 6. 
        cudaMemset(grad_d_w, 0, total_w * sizeof(float));
        cudaMemset(grad_d_a, 0, total_a * sizeof(float));
        cudaMemset(grad_wo, 0, total_wo * sizeof(float));
        cudaMemset(input_gradient_last_layer, 0, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float)); // Initialize to zero
        //memset the output gradients to zero
        for (int l = 0; l < L; ++l) {
            cudaMemset(output_gradients[l], 0, num_nodes * in_dim[l] * sizeof(float));
        }
    }

  

    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    std::cout <<  " total time: " << elapsed.count() << " ms" << std::endl;


}

