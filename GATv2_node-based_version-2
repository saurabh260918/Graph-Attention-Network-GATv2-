%%writefile cuda.cu
#include <cuda_runtime.h>
#include<cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
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


// including gradient_clip

double compute_gradient_norm(const double* grad_array, size_t size) {
    double sum_squares = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum_squares += grad_array[i] * grad_array[i];
    }
    return sqrt(sum_squares);
}




//------------------------------to copy initalised weights as text file----------------------------//

//======================TEMPORARY START====================//
void load_float_array(const std::string& filename, float** arr, int& length) {
    std::ifstream file(filename);
    std::vector<float> values;
    float val;
    while (file >> val) {
        values.push_back(val);
    }
    length = values.size();
    *arr = new float[length];
    std::copy(values.begin(), values.end(), *arr);
}

//=====================TEMPORARY END====================

void save_array_to_file(const std::string& filename, const float* array, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        file << array[i];
        if (i < size - 1) file << " ";  // Space between numbers, no space after last
    }
    file << std::endl;
    file.close();

    std::cout << "Saved " << size << " values to " << filename << std::endl;
}

//------------------------------END---------------------------------------------------------------//

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
        //printf("Value read: %d\n", val);  // Debugging line to check values read
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
    float loss = total_loss;
    float accuracy = static_cast<float>(total_correct) / N;
    printf("\nAvg Loss: %f, Accuracy: %.2f%%\n", avg_loss, 100.0f * accuracy);
    //printf("\nLoss: %f, Accuracy: %.2f%%\n", loss, 100.0f * accuracy);
    return loss;
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

__global__ void csr_to_coo_kernel(
    const int* __restrict__ d_row_ptr,  // N+1
    const int* __restrict__ d_col_idx,  // E
    int* __restrict__ d_src,            // E
    int* __restrict__ d_dst,            // E
    int N)
{
    int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst < N) {
        int start = d_row_ptr[dst];
        int end   = d_row_ptr[dst + 1];

        for (int e = start; e < end; e++) {
            d_src[e] = d_col_idx[e]; // src from col_idx
            d_dst[e] = dst;          // dst is current row
        }
    }
}



__global__ void reduce_sum_squares(const float* grad, int n, float* out) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < n) {
        float g = grad[i];
        val = g * g;
    }
    sdata[tid] = val;
    __syncthreads();

    // reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);  // CUDA >= 8 supports atomicAdd on double
    }
}



__global__ void scale_grads(float* grad, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad[i] *= scale;
    }
}
__global__ void setup_states_kernel(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, /* sequence */ idx, /* offset */ 0, &states[idx]);
}



__global__ void xavier_init_kernel_curand(
    float* d_w_src,   // [num_layers][num_heads][out_dim][in_dim]
    float* d_w_dst,   // [num_layers][num_heads][out_dim][in_dim]
    float* d_a,       // [num_layers][num_heads][out_dim]
    float* d_wo,      // [C][out_dim_last_layer]
    const int* head,      // [num_layers]
    const int* in_dim,    // [num_layers]
    const int* out_dim,   // [num_layers]
    int C,                // Number of output classes
    int num_layers,
    curandState* states   // RNG states, length >= total threads
) {
    int l = blockIdx.x;     // layer
    int h = blockIdx.y;     // head
    int o = threadIdx.x;    // output dim index

    // global thread id for RNG state
    int tid = l * gridDim.y * blockDim.x + h * blockDim.x + o;
    curandState local_state = states[tid];

    if (l < num_layers && h < head[l] && o < out_dim[l]) {
        int in_d  = in_dim[l];
        int out_d = out_dim[l];
        float limit = sqrtf(6.0f / (2 * in_d + out_d));

        // ---- compute base offsets ----
        int w_src_offset = 0;
        int w_dst_offset = 0;
        for (int i = 0; i < l; ++i) {
            w_src_offset += head[i] * out_dim[i] * in_dim[i];
            w_dst_offset += head[i] * out_dim[i] * in_dim[i];
        }
        w_src_offset += h * out_d * in_d;
        w_dst_offset += h * out_d * in_d;
        int row_offset = o * in_d;

        // ---- fill w_src[o, :] and w_dst[o, :] ----
        for (int j = 0; j < in_d; ++j) {
            float r1 = curand_uniform(&local_state);
            float v1 = r1 * 2.0f * limit - limit;
            d_w_src[w_src_offset + row_offset + j] = v1;

            float r2 = curand_uniform(&local_state);
            float v2 = r2 * 2.0f * limit - limit;
            d_w_dst[w_dst_offset + row_offset + j] = v2;
        }

        // ---- attention vector ----
        int a_offset = 0;
        for (int i = 0; i < l; ++i)
            a_offset += head[i] * out_dim[i];
        a_offset += h * out_d + o;

        float ra = curand_uniform(&local_state);
        float va = ra * 2.0f * limit - limit;
        d_a[a_offset] = va;
    }

    // ---- output projection W0 ----
    if (l == num_layers - 1 && h == 0) {
        int out_d = out_dim[l];
        if (o < out_d) {
            float limit0 = sqrtf(6.0f / (C + out_d));
            for (int r = 0; r < C; ++r) {
                float rw = curand_uniform(&local_state);
                float vw = rw * 2.0f * limit0 - limit0;
                d_wo[r * out_d + o] = vw;
            }
        }
    }

    // save state back
    states[tid] = local_state;
}


void clip_grad_norm(float* d_grad, int n, double clip_thresh) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // allocate device scalar for sum of squares
    float zero = 0.0f;
    float* d_sumsq;
    cudaMalloc(&d_sumsq, sizeof(float));
    cudaMemcpy(d_sumsq, &zero, sizeof(float), cudaMemcpyHostToDevice);

    // 1. compute total sum of squares
    reduce_sum_squares<<<blocks, threads, threads*sizeof(float)>>>(d_grad, n, d_sumsq);

    // 2. copy result back to host (just one float)
    float h_sumsq;
    cudaMemcpy(&h_sumsq, d_sumsq, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sumsq);

    float norm = sqrtf(h_sumsq);
    float scale = 1.0f;
    if (norm > clip_thresh) {
        scale = clip_thresh / (norm + 1e-9f);
    }

    // 3. apply scaling if needed
    if (scale < 1.0) {
        scale_grads<<<blocks, threads>>>(d_grad, n, scale);
    }
}


__global__ void compute_attn_scores_kernel(
    int N,                       // number of nodes
    int in_dim,                  // input feature dimension
    int out_dim,                 // output feature dimension per head
    int num_heads,               // number of attention heads
    float* d_input_features,     // SIZE: [N][in_dim] // input features to this layer
    const int* d_row_ptr,        // SIZE: [N+1]
    const int* d_col_idx,       // SIZE: [num_edges]
    const float* W_src,         // SIZE: [num_heads][out_dim][in_dim]   // weights for source node
    const float* W_dst,         // SIZE: [num_heads][out_dim][in_dim].  // weights for destination node
    const float* attn_vec,      // SIZE: [num_heads][out_dim]           // attention vector a
    float* attn_score,          // SIZE: [num_heads][E]                // output: unnormalized attention scores e_ij
    int E,                       // number of edges
    float negative_slope
){

    int node = blockIdx.x;
    int head = blockIdx.y;

    if (node >= N || head >= num_heads) return;

    extern __shared__ float shared_mem[];
    float* sh_attn_vector = shared_mem;                 // [out_dim]
    float* dst_node_features = &shared_mem[out_dim]; // [in_dim]
    int row_start = d_row_ptr[node];
    int row_end = d_row_ptr[node + 1];
    int degree = row_end - row_start;
    //fetch attn_vector and dst_node_features to shared memory
    for(int i = threadIdx.x; i < out_dim; i += blockDim.x){
        sh_attn_vector[i] = attn_vec[head * out_dim + i];
    }
    for(int i = threadIdx.x; i < in_dim; i += blockDim.x){
        dst_node_features[i] = d_input_features[node * in_dim + i];
    }
    __syncthreads();

    for(int i = threadIdx.x; i < degree; i += blockDim.x){  //each threads mostly run this loop mostly once since degree<=blockDim.x for most nodes
        int src_node = d_col_idx[row_start + i];
        float e_ij = 0.0f; // Initialize before accumulation
        for(int j = 0; j < out_dim; j++){
            float w_src_xi = 0.0f;    // W_src[j] * x_i
            float w_dst_xj = 0.0f;    // W_dst[j] * x_j
            float* src_node_features = &d_input_features[src_node * in_dim]; // [in_dim]
            for(int k = 0; k < in_dim; k++){
                // compute W_src[j] * x_i
                w_src_xi += W_src[(head * out_dim * in_dim) + (j * in_dim) + k] * src_node_features[k];
                // compute W_dst[j] * x_j
                w_dst_xj += W_dst[(head * out_dim * in_dim) + (j * in_dim) + k] * dst_node_features[k];
            } 
            // sum them
            float s_ij = w_src_xi + w_dst_xj;
            // apply leaky relu
            s_ij = leaky_relu(s_ij, negative_slope);
            // dot with attn_vector[j]
            e_ij += sh_attn_vector[j] * s_ij;
        }
        // Store the attention score for this neighbor
        attn_score[head*E + row_start + i] = e_ij;
      
    }
    

}

__global__ void compute_max_sum_attn_score(
    const int* __restrict__ d_row_ptr, 
    const float* __restrict__ attn_score, 
    int N, 
    int H, 
    int E, 
    float* __restrict__ max_score,  // [N][H]
    float* __restrict__ sum_exp)    // [N][H]
 {
    int dst = blockIdx.x;   // dst node id
    int h   = blockIdx.y;   // head id
    int lane = threadIdx.x; // 0..31

    int start = d_row_ptr[dst];
    int end   = d_row_ptr[dst + 1];

    // ---- Pass 1: Max ----
    float m = -1e9f;
    for (int e = start + lane; e < end; e += 32)  //at the end all 32 threads will have 32 max computed over all degrees
        m = fmaxf(m, attn_score[h * E + e]);

    // Warp reduce for max
    for (int offset = 16; offset > 0; offset /= 2)
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));

    m = __shfl_sync(0xffffffff, m, 0);  // broadcast

    // ---- Pass 2: Sum ----
    float s = 0.f;
    for (int e = start + lane; e < end; e += 32){
        float v = attn_score[h * E + e] - m;
        v = fmaxf(v, -80.0f);
        s += __expf(v);  // numerically stable
    }

    // Warp reduce for sum
    for (int offset = 16; offset > 0; offset /= 2)
        s += __shfl_down_sync(0xffffffff, s, offset);

    if (lane == 0) {
        max_score[N * h + dst] = m;
        sum_exp[N * h + dst] = s;
    }
}




__global__ void gatv2_forward_kernel(
    int N, int E, int in_dim, int out_dim, int num_heads,
    float* d_input_features,      // [N][in_dim]
    const int* d_row_ptr,         // [N+1]
    const int* d_col_idx,         // [num_edges]
    const float* W_src,         // [num_heads][out_dim][in_dim]
    const float* W_dst,         // [num_heads][out_dim][in_dim]
    const float* attn_vec,             // [num_heads][out_dim]
    float* d_h,                  //  [N][num_heads][out_dim] or [N][out_dim]      //pre-nonlinearity output of layers
    float* d_H,                 // [N][num_heads][out_dim] or [N][out_dim] if last layer //post-nonlinearity output of layers. it will act as input to next layer
    float* attn_score,            // [N][num_heads][max_degree]
    float* attn_coeff,            // [N][num_heads][max_degree]
    float* d_max_score,          // [N][num_heads]
    float* d_sum_exp,            // [N][num_heads]
    bool is_last_layer,
    float negative_slope
) {
    int node = blockIdx.x;
    int head = blockIdx.y;

    if (node >= N || head >= num_heads) return;
    extern __shared__ float shared_mem[];
    float* head_output = &shared_mem[2]; //[out_dim]
    int row_start = d_row_ptr[node];
    int row_end = d_row_ptr[node + 1];
    int degree = row_end - row_start;
 
    //if (threadIdx.x >= degree) return;
    for (int i = threadIdx.x; i < out_dim; i += blockDim.x)
        head_output[i] = 0.0f;
    __syncthreads();

    // offset for max/sum arrays: each node has H heads
    int offset = head * N + node;
    if(threadIdx.x == 0){
        shared_mem[0] = d_max_score[offset];
        shared_mem[1] = d_sum_exp[offset];
    }
    __syncthreads();

    // softmax for each neighbours of dst node for this head
    for(int i = threadIdx.x; i < degree; i += blockDim.x){  //each threads mostly run this loop mostly once since degree<=32 for most nodes
        int e = row_start + i;  //edge index
        float max_val = shared_mem[0];
        float sum_exp = shared_mem[1];
        // compute attention coeff
        float exp_score = attn_score[head * E + e] - max_val;
        exp_score = fmaxf(exp_score, -80.0f);  // avoid underflow
        exp_score = __expf(exp_score);  // numerically stable
        float alpha = exp_score / (sum_exp + 1e-8f);                // avoid divide by zero
        attn_coeff[e + head * E] = alpha;
    }
    __syncthreads();

    // now we compute for this destination node, for this head, the output features
    // summation of alpha_ij * W_src * x_i for all i in N(j)
    // node i is src node and node j is dst node i.e. all i's are in-degree neighbors of j



    int w_src_base = head * out_dim * in_dim;
    for(int i = threadIdx.x; i < degree; i += blockDim.x){
        int src_node = d_col_idx[row_start + i];
        float alpha = attn_coeff[head * E + row_start + i];
        float* src_node_features = &d_input_features[src_node * in_dim]; // [in_dim]
        for (int od = 0; od < out_dim; ++od) {
            float sum = 0.0f;
            for (int id = 0; id < in_dim; ++id)
                sum += W_src[w_src_base + od*in_dim + id] * src_node_features[id];
            
            atomicAdd(&head_output[od], alpha * sum);
        }
    }
    __syncthreads();

    if (!is_last_layer) {
        // add Nonlinearity then concatenate
        for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
            d_h[node * num_heads * out_dim + head * out_dim + i] = head_output[i];
            head_output[i] = leaky_relu(head_output[i]);
            d_H[node * num_heads * out_dim + head * out_dim + i] = head_output[i];
        }
    } else {
        // Final layer: average over heads
        for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
            atomicAdd(&d_h[node * out_dim + i], head_output[i] / num_heads); // Aggregated output before nonlinearity
        }
        __syncthreads();
  
        // Nonlinearity after averaging
        if (head == 0) {
            for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
                float pre_nlin = d_h[node * out_dim + i];
                float post_nlin = leaky_relu(pre_nlin);
                d_H[node * out_dim + i] = post_nlin;
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

}

__global__ void compute_output_gradients(
    const float* d_y,        // [N][C], softmax output
    const int* d_labels,     // [N],   true labels
    const float* d_h,       // [N][out_dim_L], PRE-nonlinearity (input to activation)
    const float* d_H,       // [N][out_dim_L], POST-nonlinearity (output of activation)
    const float* d_wo,       // [C][out_dim_L], output linear W
    float* grad_wo,        // [C][out_dim_L], output: grad for W_o
    float* grad_h,        // [N][num_heads][out_dim], output: grad for h_i^L (pre-activation) per head
    int N, int C, int out_dim,
    int num_heads,
    float negative_slope     // LeakyReLU slope
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;
    extern __shared__ float dL_dz_shared[];     //size [blockDim.x][C]
    //Step 1: compute dL/dz = y_hat - y
    float* dL_dz = &dL_dz_shared[threadIdx.x * C];
    for (int c = 0; c < C; ++c) {
        dL_dz[c] = d_y[node*C + c] - (c == d_labels[node] ? 1.0f : 0.0f);
    }

    // Step 2: accumulate grad for W_o: dL/dWo += dL/dz * H_i_L^T
    for (int c = 0; c < C; ++c) {
        float grad_z = dL_dz[c];
        for (int d = 0; d < out_dim; ++d) {
            float contrib = grad_z * d_H[node * out_dim + d];
            atomicAdd(&grad_wo[c*out_dim + d], contrib);
        }
        
    }

    // Step 3: backprop to H^(L): dL/dH^(L) = W_o^T * dL/dz, // store the pre-activation gradient in grad_d_hL for that node. it will be 1/num_heads of the total gradient
    float inv_heads = 1.0f / (float)num_heads;
    for (int d = 0; d < out_dim; ++d) {
        float grad_H = 0.0f;
        for (int c = 0; c < C; ++c) {
            grad_H += d_wo[c*out_dim + d] * dL_dz[c];
        }
        //grad_d_HL[node*out_dim + d] = sum;
        //grad_hL= (grad_d_HL * derivative)/num_heads. //this is per head grad


        // Step 4: Chain rule through the non-linearity (LeakyReLU)
        float h_val = d_h[node*out_dim + d];
        float derivative = (h_val > 0.0f) ? 1.0f : negative_slope;
        for (int h = 0; h < num_heads; ++h) {
            grad_h[node * num_heads * out_dim + h * out_dim + d] =
                grad_H * derivative * inv_heads;
        }
    }

}


//.........................................................................................................//

__global__ void gatv2_layer_backward(
    int N,
    int E,
    int in_dim,
    int out_dim,
    int num_heads,
    const int* d_row_ptr,         // [N+1]
    const int* d_col_idx,         // [num_edges]
    const float* d_input_features,             // [N][in_dim]
    const float* d_w_src,             // [num_heads][out_dim][in_dim]
    const float* d_w_dst,             // [num_heads][out_dim][in_dim]
    const float* d_a,               // [num_heads][out_dim]
    float* attn_coeff,        // [num_heads][E] (softmaxed attention coefficients)
    const float* input_grad,        // [N][num_heads][out_dim] (pre-nonlinearity gradients from next layer)
    float* grad_alpha,        // [num_heads][E]
    float* grad_e,            // [num_heads][E]
    float* grad_a,            // [num_heads][out_dim]
    float* grad_w_src,        // [num_heads][out_dim][in_dim]
    float* grad_w_dst,        // [num_heads][out_dim][in_dim]
    const float negative_slope

) {
    int dst_node = blockIdx.x;    // Node index
    int head = blockIdx.y;    // Head index
    int tid = threadIdx.x; // Neighbor offset
    if (dst_node >= N || head >= num_heads) return;

    //create a shared memory equal to max_degree+out_dim
    extern __shared__ float shared_memory[];    
    float* s_grad_hj = &shared_memory[0]; // size out_dim
    float* s_xj = &shared_memory[out_dim]; // size in_dim
    float* s_a = &shared_memory[out_dim + in_dim]; // size out_dim

    // Load dL/dh for this node and head into shared memory
    for(int i = threadIdx.x; i < out_dim; i += blockDim.x)
        s_grad_hj[i] = input_grad[(dst_node * num_heads + head) * out_dim + i];
    
    // Load dst node features Xj into shared memory
    for(int d = tid; d < in_dim; d += blockDim.x) {
        s_xj[d] = d_input_features[dst_node * in_dim + d];
    }

    // Load attention vector a^h
    for(int d = tid; d < out_dim; d += blockDim.x) {
        s_a[d] = d_a[head * out_dim + d];
    }

   __syncthreads();

    int row_start = d_row_ptr[dst_node];
    int row_end = d_row_ptr[dst_node + 1];
    int deg = row_end - row_start;
    if (deg == 0) return;

    int j = row_start + tid;
    int src_node = d_col_idx[j]; // Neighbor node index
    float* w_src_base = (float*)&d_w_src[head * out_dim * in_dim]; // [out_dim][in_dim]

    // --- Step D.2: dL/d alpha_ij ---
    for(int i = threadIdx.x; i < deg; i += blockDim.x) {
        float dL_d_alpha_ij = 0.0f;
        for (int od = 0; od < out_dim; ++od) {
            //float dL_d_h = d_higher[(i * num_heads + h) * out_dim + od];
            for (int id = 0; id < in_dim; ++id)
                dL_d_alpha_ij += s_grad_hj[od] * w_src_base[od * in_dim + id] * d_input_features[src_node * in_dim + id];   
        }
        // store dL_d_alpha_ij in global memory
        grad_alpha[head * E + row_start + i] = dL_d_alpha_ij;
    }
    __syncthreads();


    // --- Step E.1: dL/d e_ij ---
    for(int i = threadIdx.x; i < deg; i += blockDim.x) {
        float dL_d_e_ij = 0.0f;
        float alpha_ij = attn_coeff[E * head + row_start + i];
        for (int k = 0; k < deg; ++k) {
            float alpha_kj = attn_coeff[E * head + row_start + k];
            float dL_d_alpha_kj = grad_alpha[head * E + row_start + k];
            if (k == i) {
                // ∂α_ij/∂e_ij when k=i: α_ij * (1 - α_ij)
                dL_d_e_ij += dL_d_alpha_kj * alpha_ij * (1.0f - alpha_ij);
            } else {
                // ∂α_kj/∂e_ij when k≠i: -α_kj * α_ij
                dL_d_e_ij += dL_d_alpha_kj * (-alpha_kj * alpha_ij);
            }
        }
        //load into global memory
        grad_e[head * E + row_start + i] = dL_d_e_ij;
    }
    __syncthreads();

    // Base offset for this head's weight matrices
    int w_offset = head * out_dim * in_dim;
    
    // Process edges in parallel with strided loop
    for(int i = tid; i < deg; i += blockDim.x) {
        int edge_idx = row_start + i;
        int src = d_col_idx[edge_idx];
        
        float grad_e_ij = grad_e[head * E + edge_idx];
        float alpha_ij = grad_alpha[head * E + edge_idx];
        
        // Process each output dimension
        for(int d = 0; d < out_dim; d++) {
            // Compute S_ij^{l,h}[d] = W_src[d,:] * xi + W_dst[d,:] * xj (on-the-fly)
            float sij_d = 0.0f;
            for(int k = 0; k < in_dim; k++) {
                float xi_k = d_input_features[src * in_dim + k];
                float w_src_dk = d_w_src[w_offset + d * in_dim + k];
                float w_dst_dk = d_w_dst[w_offset + d * in_dim + k];
                sij_d += w_src_dk * xi_k;
                sij_d += w_dst_dk * s_xj[k];
            }
            
            // Compute Leaky'(S_ij^{l,h}[d]) and Leaky(S_ij^{l,h}[d]) on-the-fly
            float leaky_prime_sij_d = (sij_d > 0.0f) ? 1.0f : negative_slope;
            float leaky_sij_d = (sij_d > 0.0f) ? sij_d : negative_slope * sij_d;
            
            // ===== Compute gradient w.r.t. attention vector a =====
            // ∂L/∂a = Σ_j Σ_{i∈N_j} (∂L/∂e_ij^{l,h} · Leaky(S_ij^{l,h}))
            float contrib_a = grad_e_ij * leaky_sij_d;
            atomicAdd(&grad_a[head * out_dim + d], contrib_a);
            
            // Compute a^h[d] ⊙ Leaky'(S_ij^{l,h}[d])
            float a_leaky_prime = s_a[d] * leaky_prime_sij_d;
            
            // Process each input dimension for weight gradients
            for(int k = 0; k < in_dim; k++) {
                float xi_k = d_input_features[src * in_dim + k];
                
                // ===== Compute gradient w.r.t. W_src (direct path) =====
                // ∂L/∂W_src|_direct = Σ_j [∂L/∂h_j^{l+1,h} · Σ_{i∈N_j} α_ij^{l,h} · xi]
                float contrib_w_src_direct = s_grad_hj[d] * alpha_ij * xi_k;
                
                // ===== Compute gradient w.r.t. W_src (attention path) =====
                // ∂L/∂W_src|_atten = Σ_j Σ_{i∈N_j} (∂L/∂e_ij^{l,h} [a^h ⊙ Leaky'(S_ij^{l,h})] · xi)
                float contrib_w_src_atten = grad_e_ij * a_leaky_prime * xi_k;
                
                // Total contribution to W_src (both paths)
                float contrib_w_src = contrib_w_src_direct + contrib_w_src_atten;
                atomicAdd(&grad_w_src[w_offset + d * in_dim + k], contrib_w_src);
                
                // ===== Compute gradient w.r.t. W_dst =====
                // ∂L/∂W_dst = Σ_j Σ_{i∈N_j} (∂L/∂e_ij^{l,h} [a^h ⊙ Leaky'(S_ij^{l,h})] · xj)
                float contrib_w_dst = grad_e_ij * a_leaky_prime * s_xj[k];
                atomicAdd(&grad_w_dst[w_offset + d * in_dim + k], contrib_w_dst);
            }
        }
    } 

}

__global__ void compute_features_input_gradients(
    int N,
    int H,
    int E,
    int in_dim,
    int out_dim,
    float negative_slope,
    const int* src_idx,                // [E]
    const int* dst_idx,                // [E]
    const float* attn_coeff,           // [H][E]
    const float* input_features,      // [N][in_dim]
    const float* d_w_src,             // [H][out_dim][in_dim]
    const float* d_w_dst,             // [H][out_dim][in_dim]
    const float* d_input_gradients,   // [N][H][out_dim]     //gradient flowing from upper layer to this layer. act as input gradient to this layer.
    const float* grad_attn_score,     // [H][E]
    const float* attn_vector,         // [H][out_dim]
    float* grad_x_features            // [N][in_dim]
) {

    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(edge_id >= E) return;
    int src = src_idx[edge_id];
    int dst = dst_idx[edge_id];
    extern __shared__ float shmem[]; // size: out_dim
    float* sh_a = &shmem[0];          // [size: out_dim]
    // loop over heads

    for (int h = 0; h < H; ++h) {
        const float* a_h = attn_vector + h * out_dim;                        // [out_dim]
        //fill sh_a from a_h using multiple threads
        for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
            sh_a[i] = a_h[i];
        }
        __syncthreads();
        int base = h * out_dim * in_dim;         // [out_dim][in_dim]
        float  dl_de = grad_attn_score[h * E + edge_id];                 // ∂L/∂e_{ij}^{h}
        float  alpha_ij = attn_coeff[h * E + edge_id];                   // α_{ij}^{h}
        const float* x_src = input_features + (src * in_dim);    // x_i
        const float* x_dst = input_features + (dst * in_dim);    // x_j
        const float* grad_dst = d_input_gradients + (dst * H + h) * out_dim; // ∂L/∂h_j^{h}
        float* grad_x_src = grad_x_features + (src * in_dim); // ∂L/∂x_i
        float* grad_x_dst = grad_x_features + (dst * in_dim); // ∂L/∂x_j
        // here each thread working on each edge will compute the contribution to input feature gradient of src and dst node
        // loop over output dimension
        for (int od = 0; od < out_dim; ++od) {
            // Compute S_ij^{h}[od] = W_src[od,:] * xi + W_dst[od,:] * xj (on-the-fly)
            float sij_od = 0.0f;
            for (int id = 0; id < in_dim; ++id) {
                float w_src_od_id = d_w_src[base + od * in_dim + id];
                float w_dst_od_id = d_w_dst[base + od * in_dim + id];
                sij_od += w_src_od_id * x_src[id];
                sij_od += w_dst_od_id * x_dst[id];
            }
            // Compute Leaky'(S_ij^{h}[od]) on the-fly
            float leaky_prime_sij_od = (sij_od > 0.0f) ? 1.0f : negative_slope;
            // Compute a^h[od] ⊙ Leaky'(S_ij^{h}[od])
            float a_leaky_prime = sh_a[od] * leaky_prime_sij_od;
            // Contribution to ∂L/∂x_i (src node)
            for (int id = 0; id < in_dim; ++id) {
                // Direct path contribution ∂L/∂x_src|_direct = Σ_h [∂L/∂h_dst^{h} · α_ij^{h} · W_src[:,id]]
                // Attention path contribution ∂L/∂x_src|_atten = Σ_h (∂L/∂e_ij^{h} [a^h ⊙ Leaky'(S_ij^{h})] · W_src[:,id])
                // Attention path contribution ∂L/∂x_dst = Σ_h (∂L/∂e_ij^{h} [a^h ⊙ Leaky'(S_ij^{h})] · W_dst[:,id])
                float w_src_od_id = d_w_src[base + od * in_dim + id];
                float w_dst_od_id = d_w_dst[base + od * in_dim + id];
                float contrib_x_src = grad_dst[od] * alpha_ij * w_src_od_id
                                    + dl_de * a_leaky_prime * w_src_od_id;
                float contrib_x_dst = dl_de * a_leaky_prime * w_dst_od_id;
                atomicAdd(&grad_x_src[id], contrib_x_src);
                atomicAdd(&grad_x_dst[id], contrib_x_dst);
            }

        }
    }
}

// now i have gradient of input feature for each node. this i will pass through activation function 
// and get the gradient of input feature before activation and this preactivation gradient will serve
// as input gradient to previous layer.
__global__ void compute_preActivation_inputFeatures_gradient(
    int N,
    float negative_slope,
    int in_dim,
    const float* pre_activation_input_features, // [N][in_dim], input features before activation
    float* input_features_gradients // [N][in_dim], gradient of input features after activation
) {
    int nodeId = blockIdx.x * blockDim.x + threadIdx.x;
    if (nodeId >= N) return;
    for(int d=0; d<in_dim; d++){
            float post_activation_gradients = input_features_gradients[(nodeId * in_dim + d)];
            float derivative = (pre_activation_input_features[(nodeId * in_dim + d)] > 0.0f) ? 1.0f : negative_slope;
            input_features_gradients[(nodeId * in_dim + d)] = post_activation_gradients * derivative;
    }
}

__global__ void adam_update_kernel(
    float* params,     // current weights (model parameters)
    const float* grads,// gradients of loss w.r.t. weights
    float* m,          // first moment (moving average of gradients)
    float* v,          // second moment (moving average of squared gradients)
    float lr,          // learning rate
    size_t n,          // number of parameters
    float beta1,       // momentum term for m (usually 0.9)
    float beta2,       // decay term for v (usually 0.999)
    float epsilon,     // small constant to prevent division by 0 (1e-8)
    int t              // current time step (epoch or iteration)
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];

        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1.0f - beta2) * (grads[i] * grads[i]);

        // Compute bias-corrected first moment estimate
        float m_hat = m[i] / (1.0f - powf(beta1, t));

        // Compute bias-corrected second raw moment estimate
        float v_hat = v[i] / (1.0f - powf(beta2, t));

        // Update parameters
        params[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}


__global__ void sgd_update_kernel(float* params, const float* grads, float lr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        params[i] -= lr * (float)grads[i];
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
    //float prob = fmaxf(d_y[idx * C + label], 1e-10f); // avoid log(0)
    float prob = d_y[idx * C + label];
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




int main(int argc, char** argv) {
    // Measure GPU memory before allocations
    size_t free_before, total_mem;
    cudaMemGetInfo(&free_before, &total_mem);
    printf("\n[Memory Tracker] Before allocation:\n");
    printf("  Total GPU memory: %.2f MB\n", total_mem / (1024.0 * 1024.0));
    printf("  Free GPU memory : %.2f MB\n", free_before / (1024.0 * 1024.0));
    // Default hyperparameters
    int epochs = 200;
    int L = 2;
    bool clip = false;
    std::string optimizer = "sgd";
    float lr = 0.0001f, beta1 = 0.9f, beta2 = 0.999f;
    bool beta1_specified = false, beta2_specified = false;

    // Parse command-line args to get L first
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--num-layers" && i + 1 < argc) {
            L = std::stoi(argv[++i]);
            if (L <= 0) {
                std::cerr << "Error: Number of layers must be > 0\n";
                return 1;
            }
            break;
        }
    }
    int* head       = new int[L];
    int* out_dim    = new int[L];
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--heads" && i + 1 < argc) {
            std::string val = argv[++i];
            std::stringstream ss(val);
            std::string item;
            for (int l = 0; l < L; ++l) {
                if (!std::getline(ss, item, ',')) {
                    std::cerr << "Error: --heads must have " << L << " values.\n";
                    return 1;
                }
                head[l] = std::stoi(item);
            }
        }
        else if (arg == "--outdims" && i + 1 < argc) {
            std::string val = argv[++i];
            std::stringstream ss(val);
            std::string item;
            for (int l = 0; l < L; ++l) {
                if (!std::getline(ss, item, ',')) {
                    std::cerr << "Error: --ooutdims must have " << L << " values.\n";
                    return 1;
                }
                out_dim[l] = std::stoi(item);
            }
        }
        else if (arg == "--clip") {
            clip = true;
        }
        else if (arg == "--optimizer" && i + 1 < argc) {
            optimizer = argv[++i];
            if(optimizer != "sgd" && optimizer != "adam") {
                std::cerr << "Invalid optimizer choice. Use 'sgd' or 'adam'\n";
                return 1;
            }
        }
        // Parse args (optional)
        else if (arg == "--beta1" && i + 1 < argc){
            beta1 = std::stof(argv[++i]);
            beta1_specified = true;
        }
        else if (arg == "--beta2" && i + 1 < argc) {
            beta2 = std::stof(argv[++i]);
            beta2_specified = true;
        }
        else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        }
    }
    if (optimizer == "adam") {
        if (beta1 <= 0.0f || beta1 >= 1.0f || beta2 <= 0.0f || beta2 >= 1.0f) {
            std::cerr << "Error: For Adam optimizer, beta1 and beta2 must be in (0,1).\n";
            return 1;
        }
    } else if (optimizer == "sgd") {
            if (beta1_specified || beta2_specified) {
                std::cerr << "Warning: beta1/beta2 specified but ignored for SGD optimizer.\n";
            }
    }


    // Print configuration
    std::cout << "Configuration:\n"
              << "  Number of layers: " << L << "\n"
              << "  Epochs: " << epochs << "\n"
              << "  Attention heads: [";
    for (int l = 0; l < L; ++l) {
        std::cout << head[l];
        if (l < L-1) std::cout << ", ";
    }
    std::cout << "]\n  Output dimensions: [";
    for (int l = 0; l < L; ++l) {
        std::cout << out_dim[l];
        if (l < L-1) std::cout << ", ";
    }
    std::cout << "]\n"
              << "  Gradient clipping: " << (clip ? "true" : "false") << "\n"
              << "  Optimizer: " << optimizer << "\n"
              << "  Learning rate: " << lr << "\n\n";

    // 1. Load graph in CSR format
    int num_nodes, num_edges, input_dim;
    float* h_features;    // [num_nodes][input_dim]
    int* h_row_ptr;       // [num_nodes+1]
    int* h_col_idx;       // [num_edges]
    int* h_labels;       // [num_nodes]

    // Load features
    std::string dataset_name = "pubmed";
    std::string data_root = "./data";
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            dataset_name = argv[++i];
        } 
        else if (arg == "--data-root" && i + 1 < argc) {
            data_root = argv[++i];
        }
    }

    // Fallback: check environment variable (used if --data-root not provided)
    const char* env_root = std::getenv("DATA_ROOT");
    if (env_root && data_root == "./data") { // only override if user didn't pass CLI flag
        data_root = std::string(env_root);
    }

    // Ensure trailing slash
    if (!data_root.empty() && data_root.back() != '/' && data_root.back() != '\\')
        data_root += '/';

    std::string dataset_path = data_root + dataset_name + "/";


    std::cout << "Using dataset: " << dataset_name << std::endl;
    std::cout << "Dataset path: " << dataset_path << std::endl;

    load_features((dataset_path + "features.txt").c_str(), &h_features, num_nodes, input_dim);

    // Load row_ptr
    int row_ptr_len;
    load_int_array((dataset_path + "row_ptr.txt").c_str(), &h_row_ptr, row_ptr_len);
    if (row_ptr_len != num_nodes + 1) {
        std::cerr << "Invalid row_ptr length\n";
        return 1;
    }

    // Load col_idx
    int col_idx_len;
    load_int_array((dataset_path + "col_idx.txt").c_str(), &h_col_idx, col_idx_len);
    num_edges = col_idx_len;

    //load labels
    int labels_len;
    load_int_array((dataset_path + "labels.txt").c_str(), &h_labels, labels_len);
    if (labels_len != num_nodes) {
        std::cerr << "Invalid labels length\n";
        return 1;
    }
    


    int max_degree = compute_max_degree(h_row_ptr, num_nodes);
    std::cout << "Max degree = " << max_degree << std::endl;

    auto itr = thrust::max_element(thrust::host, h_labels, h_labels + num_nodes);
    int C = *itr + 1; // Number of classes, assuming labels are 0-indexed
    std::cout << "Number of classes = " << C << std::endl;



    int* in_dim = new int[L]; // Input dim for first layer, subsequent layers will be computed based on previous layer's output.
    in_dim[0] = input_dim;
    for (int l = 1; l < L; ++l)
        in_dim[l] = head[l-1] * out_dim[l-1];

    int max_heads = *std::max_element(head, head + L);  // Maximum number of heads across all layers
    int max_out_dim = *std::max_element(out_dim, out_dim + L); // Maximum output dimension across all layers

    // 3. Declare device pointers for all parameters and caches
    int* d_head;             // Device array for number of attention heads per layer
    int* d_out_dim;         // Device array for output dimensions per layer
    int* d_in_dim;          // Device array for input dimensions per layer
    float* d_w_src;         // flat Weight matrices array for all layer
    float* d_w_dst;          // flat Weight matrices array for all layer
    float* d_attn_vec;           // flat Attention vectors array for all layers
    float* d_input_features;      // Device input features
    int* d_row_ptr;         // Device CSR row pointer
    int* d_col_idx;           // Device CSR edge array
    int* d_labels;           // Device labels array
    float** d_H = new float*[L];          // Output buffers per layer post-nonlinearity
    float** d_h = new float*[L];            // pre-nonlinearity output of hidden layers
    float* d_wo;            // Device linear transformation weight matrix of size C X out_dim.
    float* d_z;        // output after linear transformation of size N X C.
    float* d_y;        // output probabilities [N][C]
    float** d_attn_score = new float*[L]; // Attention scores per layer [num_heads][edges]
    float** d_attn_coeff = new float*[L]; // Attention coefficients per layer [num_heads][edges]
    float* d_grad_wo;    // [C][out_dim_last_layer]
    float* d_grad_w_src;   // [total_w]
    float* d_grad_w_dst;  // [total_w]
    float* d_grad_attn_vec;   // [total_a]
    float  negative_slope = 0.01f; // LeakyReLU slope
    float* d_loss;      // [num_nodes]
    int*   d_correct;   // [num_nodes]
    int* d_src;        // COO source array [num_edges]
    int* d_dst;       // COO destination array [num_edges]
    size_t total_w, total_a; // Total sizes for weights and attention vectors
    int* wt_offset = new int[L];   // Offset for each layer's weights in the flat array
    int* attn_offset = new int[L]; // Offset for each layer's attention vectors in the flat array
    curandState* d_states; // for xavier initialization of weights
    size_t total_input_grad_size; // Total size for input gradients across all layers
    float** input_gradients = new float*[L];     // Array of pointers to output gradients per layer//these are the gradients which are coming out of that layer which will become as input gradient to previous layer
    float* grad_attn_score; // Gradient of attention scores for edges  // these we will utilise at each layer during backprop.
    float* grad_attn_coeff; // Gradient of attention coefficients for edges
    float* max_score;      // [max_heads][num_nodes]
    float* sum_exp_score;  // [max_heads][num_nodes]
    int nthreads = L * max_heads * max_out_dim;


    // Compute total size needed for all layers learnable parameters
    total_w = 0;
    total_a = 0;
    wt_offset[0]= 0;   // it will have how much size consumed till previous layer
    attn_offset[0]= 0; // it will have how much size consumed till previous layer

    for (int l = 1; l < L; ++l) {
        wt_offset[l] = wt_offset[l-1] + head[l-1] * out_dim[l-1] * in_dim[l-1];
        attn_offset[l] = attn_offset[l-1] + head[l-1] * out_dim[l-1];
        total_w += head[l-1] * out_dim[l-1] * in_dim[l-1];
        total_a += head[l-1] * out_dim[l-1];
    }
    total_w += head[L-1] * out_dim[L-1] * in_dim[L-1]; // Last layer weights
    total_a += head[L-1] * out_dim[L-1]; // Last layer attention vectors

    printf("\n Total size of learnable parameters for all layers: %zu KB\n", (total_w+ total_a + (C*out_dim[L-1]) * sizeof(float)) / (1024));
    // Start timer
    // auto start = std::chrono::high_resolution_clock::now();

     // 4. Initialize parameters and allocate memory
    cudaError_t err;
    // 1. Malloc and memcpy for graph data
    err = cudaMalloc(&d_input_features, num_nodes * in_dim[0] * sizeof(float));
    //printf("cudaMalloc d_features: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(d_input_features, h_features, num_nodes * in_dim[0] * sizeof(float), cudaMemcpyHostToDevice);
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
    cudaMalloc(&d_loss,    num_nodes * sizeof(float));
    //printf("cudaMalloc d_loss: %s\n", cudaGetErrorString(err));
    cudaMalloc(&d_correct, num_nodes * sizeof(int));
    //printf("cudaMalloc d_correct: %s\n", cudaGetErrorString(err));
    cudaMalloc(&d_src, num_edges * sizeof(int));
    //printf("cudaMalloc d_src: %s\n", cudaGetErrorString(err));
    cudaMalloc(&d_dst, num_edges * sizeof(int));
    //printf("cudaMalloc d_dst: %s\n", cudaGetErrorString(err));
    printf("\nsize of graph data (CSR,COO, labels) : %zu MB\n", (((num_nodes+1) + num_edges + num_nodes + 2* num_edges)* sizeof(int)) / (1024 * 1024));
    

    for (int l = 0; l < L; ++l) {
        size_t output_size;
        if (l == L - 1) {
            // Last layer: average heads, only out_dim[l] per node
            output_size = num_nodes * out_dim[l];
        } else {
            // Non-last layer: concatenate heads
            output_size = num_nodes * head[l] * out_dim[l];
        }

        err = cudaMalloc(&d_H[l], output_size * sizeof(float));
        //printf("cudaMalloc d_H[%d]: %s\n", l, cudaGetErrorString(err));
        err = cudaMalloc(&d_h[l], output_size * sizeof(float));
        //printf("cudaMalloc d_h[%d]: %s\n", l, cudaGetErrorString(err));

        err = cudaMalloc(&d_attn_score[l], head[l] * num_edges * sizeof(float));
        //printf("cudaMalloc d_attn_score[%d]: %s\n", l, cudaGetErrorString(err));
        err = cudaMalloc(&d_attn_coeff[l], head[l] * num_edges * sizeof(float));
        //printf("cudaMalloc d_attn_coeff[%d]: %s\n", l, cudaGetErrorString(err));
    }
    // 2. Malloc for all parameters and gradients

    cudaMalloc(&d_w_src, total_w * sizeof(float));
    cudaMalloc(&d_grad_w_src, total_w * sizeof(float));
    cudaMalloc(&d_w_dst, total_w * sizeof(float));
    cudaMalloc(&d_grad_w_dst, total_w * sizeof(float));
    cudaMalloc(&d_attn_vec, total_a * sizeof(float));
    cudaMalloc(&d_grad_attn_vec, total_a * sizeof(float));
    cudaMalloc(&d_wo, C * out_dim[L - 1] * sizeof(float));
    cudaMalloc(&d_grad_wo, C * out_dim[L - 1] * sizeof(float));

    cudaMemset(d_grad_wo, 0, C * out_dim[L - 1] * sizeof(float)); // Initialize to zero
    cudaMemset(d_grad_w_src, 0, total_w * sizeof(float)); // Initialize to zero
    cudaMemset(d_grad_w_dst, 0, total_w * sizeof(float)); // Initialize to zero
    cudaMemset(d_grad_attn_vec, 0, total_a * sizeof(float)); // Initialize to zero

    cudaMalloc(&d_z, num_nodes * C * sizeof(float));
    cudaMalloc(&d_y, num_nodes * C * sizeof(float));
    cudaMalloc(&d_states, nthreads * sizeof(curandState));  // CURAND states for Xavier init

    total_input_grad_size = 0;
    for (int l = L-1; l >= 0; --l) {
        size_t input_grad_size = (size_t)num_nodes * head[l] * out_dim[l] * sizeof(float);
        total_input_grad_size += input_grad_size;
        cudaMalloc(&input_gradients[l], input_grad_size);
        cudaMemset(input_gradients[l], 0, input_grad_size); // Initialize to zero
    }
    cudaMalloc(&grad_attn_score, max_heads * num_edges * sizeof(float));
    cudaMalloc(&grad_attn_coeff, max_heads * num_edges * sizeof(float));
    cudaMalloc(&max_score, max_heads * num_nodes * sizeof(float));
    cudaMalloc(&sum_exp_score, max_heads * num_nodes * sizeof(float));
    printf("Total size of intermediate computation memory: %zu MB\n", ((total_input_grad_size + max_heads * num_edges * 2 + max_heads * num_nodes * 2) * sizeof(float)) / (1024 * 1024));


    //==========================================================
    //=============INITIALISE ADAM PARAMETERS========================
    float *m_w_src;
    cudaMalloc(&m_w_src, total_w * sizeof(float));
    cudaMemset(m_w_src, 0, total_w * sizeof(float));
    float *v_w_src;
    cudaMalloc(&v_w_src, total_w * sizeof(float));
    cudaMemset(v_w_src, 0, total_w * sizeof(float));

    float *m_w_dst;
    cudaMalloc(&m_w_dst, total_w * sizeof(float));
    cudaMemset(m_w_dst, 0, total_w * sizeof(float));
    float *v_w_dst;
    cudaMalloc(&v_w_dst, total_w * sizeof(float));
    cudaMemset(v_w_dst, 0, total_w * sizeof(float));

    float *m_a;
    cudaMalloc(&m_a, total_a * sizeof(float));
    cudaMemset(m_a, 0, total_a * sizeof(float));
    float *v_a;
    cudaMalloc(&v_a, total_a * sizeof(float));
    cudaMemset(v_a, 0, total_a * sizeof(float));

    int total_wo = C * out_dim[L - 1];
    float *m_wo;
    cudaMalloc(&m_wo, total_wo * sizeof(float));
    cudaMemset(m_wo, 0, total_wo * sizeof(float));
    float *v_wo;
    cudaMalloc(&v_wo, total_wo * sizeof(float));
    cudaMemset(v_wo, 0, total_wo * sizeof(float));
    //================Initialize parameters using Xavier initialization=========================
    // 1. Setup CURAND states
   
    int blockSize = 128;
    dim3 gridSize((nthreads + blockSize - 1) / blockSize);
    unsigned long seed = static_cast<unsigned long>(time(NULL));
    setup_states_kernel<<<gridSize, blockSize>>>(d_states, seed);
    cudaDeviceSynchronize();
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching setup_states_kernel: %s\n", cudaGetErrorString(err));
    }


    // 2. Call Xavier initialization kernel
    dim3 grid(L, max_heads);
    int block = max_out_dim;      //ensuure to be less than 1024

    xavier_init_kernel_curand<<<grid, block>>> (d_w_src, d_w_dst, d_attn_vec, d_wo, d_head, d_in_dim, d_out_dim, C, L, d_states);
    cudaDeviceSynchronize();

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching xavier_init_kernel: %s\n", cudaGetErrorString(err));
    }

    //==================CSR TO COO===========================
    int threads_coo = 256;
    int blocks_coo = (num_nodes + threads_coo - 1) / threads_coo;

    csr_to_coo_kernel<<<blocks_coo, threads_coo>>>(d_row_ptr, d_col_idx, d_src, d_dst, num_nodes);
    cudaDeviceSynchronize();
    //check for error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching csr_to_coo_kernel: %s\n", cudaGetErrorString(err));
    }    
    //==================END===========================================

    // Measure GPU memory after all allocations
    size_t free_after;
    cudaMemGetInfo(&free_after, &total_mem);
    double used_mb = (double)(free_before - free_after) / (1024.0 * 1024.0);

    printf("\n[Memory Tracker] After all allocations:\n");
    printf("  Free GPU memory : %.2f MB\n", free_after / (1024.0 * 1024.0));
    printf("  Approx. GPU memory allocated by this program: %.2f MB\n", used_mb);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        printf("\nEpoch %d\n", epoch);
        auto start = std::chrono::high_resolution_clock::now();
        // b. Forward pass for each layer
        float* d_layer_inputs = d_input_features;
        for (int l = 0; l < L; ++l) {
            dim3 grid(num_nodes, head[l]);
            int block = 32;
            size_t shared_mem = (out_dim[l] + in_dim[l]) * sizeof(float);
            bool is_last_layer = (l == L - 1);
            const float* d_w_src_l = d_w_src + wt_offset[l];
            const float* d_w_dst_l = d_w_dst + wt_offset[l];
            const float* d_a_l = d_attn_vec + attn_offset[l];

            compute_attn_scores_kernel<<<grid, block, shared_mem>>>(
                num_nodes, in_dim[l], out_dim[l], head[l], d_layer_inputs,
                d_row_ptr, d_col_idx, d_w_src_l, d_w_dst_l, d_a_l, d_attn_score[l], num_edges, negative_slope
            );
            cudaDeviceSynchronize();
            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching compute_attn_scores_kernel: %s\n", cudaGetErrorString(err));
            }

            compute_max_sum_attn_score<<<grid, block>>>(
                d_row_ptr, d_attn_score[l], num_nodes, head[l], num_edges, max_score, sum_exp_score
            );
            cudaDeviceSynchronize();
            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching compute_max_sum_attn_score: %s\n", cudaGetErrorString(err));
            }
            size_t shared_memory= (out_dim[l]+2) * sizeof(float);

            gatv2_forward_kernel<<<grid, block, shared_memory>>>(
                num_nodes, num_edges, in_dim[l], out_dim[l], head[l], d_layer_inputs,
                d_row_ptr, d_col_idx, d_w_src_l, d_w_dst_l, d_a_l, d_h[l], d_H[l], d_attn_score[l], d_attn_coeff[l],
                max_score, sum_exp_score, is_last_layer, negative_slope
            );
            cudaDeviceSynchronize();

            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching gatv2_forward_kernel: %s\n", cudaGetErrorString(err));
            }

            d_layer_inputs = d_H[l]; // Output of current layer becomes input to next


        }

        int threads_per_block = 128;
        int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

        gatv2_output_kernel<<<num_blocks, threads_per_block>>>(d_wo,  d_H[L-1], d_z, d_y, num_nodes, C, out_dim[L - 1]);
        cudaDeviceSynchronize();
        // Check for errors after gatv2_output_kernel
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error launching gatv2_output_kernel: %s\n", cudaGetErrorString(err));
        }

        
        // c. Calculate loss and accuracy ----

        compute_loss_accuracy_kernel<<<num_blocks, threads_per_block>>>(d_y, d_labels, d_loss, d_correct, num_nodes, C);
        cudaDeviceSynchronize();

        compute_loss_and_accuracy(num_nodes, d_loss, d_correct);
        cudaDeviceSynchronize();
        // d. Backward pass
        // 1: Output layer gradient
        size_t shared_memory = (C) * threads_per_block * sizeof(float);
        compute_output_gradients<<<num_blocks, threads_per_block, shared_memory>>>(
            d_y, d_labels, d_h[L-1], d_H[L-1], d_wo, d_grad_wo,
            input_gradients[L-1], num_nodes, C, out_dim[L-1], head[L-1], negative_slope
        );
        cudaDeviceSynchronize();
        // Check for errors after compute_output_gradients
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error launching compute_output_gradients: %s\n", cudaGetErrorString(err));
        }


        // 2: Loop backward through GAT layers
        for (int l = L-1; l >= 0; l--) {
            //printf("\nBackward pass for layer %d\n", l);
            dim3 grid(num_nodes, head[l]);
            int block = 32;
            size_t shared_mem = (in_dim[l] + 3 * out_dim[l]) * sizeof(float);
            float* d_w_src_l= d_w_src + wt_offset[l];
            float* d_w_dst_l = d_w_dst + wt_offset[l];
            float* d_a_l = d_attn_vec + attn_offset[l];

            gatv2_layer_backward<<<grid, block, shared_mem>>>(
            num_nodes, num_edges, in_dim[l], out_dim[l], head[l], d_row_ptr, d_col_idx,
            (l > 0) ? d_H[l - 1] : d_input_features, d_w_src_l, d_w_dst_l, d_a_l, d_attn_coeff[l],
            input_gradients[l], grad_attn_score, grad_attn_coeff, d_grad_attn_vec + attn_offset[l],
            d_grad_w_src + wt_offset[l], d_grad_w_dst + wt_offset[l], negative_slope
            );
            cudaDeviceSynchronize();
            // Check for errors after gatv2_layer_backward
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching gatv2_layer_backward: %s\n", cudaGetErrorString(err));
                return 1;
            }

            if(l == 0) break;

            int threads_grad_input = 256;
            int blocks_grad_input = (num_edges + threads_grad_input - 1) / threads_grad_input;
            float shared_mem_input = out_dim[l] * sizeof(float);
            compute_features_input_gradients<<<blocks_grad_input, threads_grad_input, shared_mem_input>>>(
                num_nodes, head[l], num_edges, in_dim[l], out_dim[l],
                negative_slope, d_src, d_dst, d_attn_coeff[l], d_H[l-1], d_w_src + wt_offset[l], d_w_dst + wt_offset[l],
                input_gradients[l], grad_attn_score, d_attn_vec + attn_offset[l], input_gradients[l-1]
            );
            cudaDeviceSynchronize();
            // Check for errors after compute_features_input_gradients
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching compute_features_input_gradients: %s\n", cudaGetErrorString(err));
            }
            int numThreads_preActGrad = 256;
            int numBlocks_preActGrad = (num_nodes + numThreads_preActGrad - 1) / numThreads_preActGrad;
            compute_preActivation_inputFeatures_gradient<<<numBlocks_preActGrad, numThreads_preActGrad>>>(
                num_nodes, negative_slope, in_dim[l], d_h[l-1], input_gradients[l-1]
            );
            cudaDeviceSynchronize();
            // Check for errors after compute_preActivation_inputFeatures_gradient
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching compute_preActivation_inputFeatures_gradient: %s\n", cudaGetErrorString(err));
            }

        }

        // // -----> *** 5. PARAMETER UPDATE SECTION —  ***
            if(clip){
                //---------_CLIP GRADIENTS___________________
                clip_grad_norm(d_grad_w_src, total_w, 5.0f);   //  threshold=5
                clip_grad_norm(d_grad_w_dst, total_w, 5.0f);
                clip_grad_norm(d_grad_attn_vec, total_a, 5.0f);
                clip_grad_norm(d_grad_wo, C * out_dim[L - 1], 5.0f);
            }
            
        int block_size = 256;

        if(optimizer == "adam"){
            // =====================ADAM UPDATE=========================
            //--- Update d_w_src ---
            int num_blocks_w = (total_w + block_size - 1) / block_size;
            adam_update_kernel<<<num_blocks_w, block_size>>>(d_w_src, d_grad_w_src, m_w_src, v_w_src, lr, total_w, beta1, beta2, 1e-8f, epoch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching adam_update_kernel for d_w_src: %s\n", cudaGetErrorString(err));
            }
            // --- Update d_w_dst ---
            adam_update_kernel<<<num_blocks_w, block_size>>>(d_w_dst, d_grad_w_dst, m_w_dst, v_w_dst, lr, total_w, beta1, beta2, 1e-8f, epoch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching adam_update_kernel for d_w_dst: %s\n", cudaGetErrorString(err));
            }

            // --- Update attention vectors d_a ---
            int num_blocks_a = (total_a + block_size - 1) / block_size;
            adam_update_kernel<<<num_blocks_a, block_size>>>(d_attn_vec, d_grad_attn_vec, m_a, v_a, lr, total_a, beta1, beta2, 1e-8f, epoch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching adam_update_kernel for d_attn_vec: %s\n", cudaGetErrorString(err));
            }

            // --- Update output weights d_wo ---
            int total_wo = C * out_dim[L - 1];
            int num_blocks_wo = (total_wo + block_size - 1) / block_size;
            adam_update_kernel<<<num_blocks_wo, block_size>>>(d_wo, d_grad_wo, m_wo, v_wo, lr, total_wo, beta1, beta2, 1e-8f, epoch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching adam_update_kernel for d_wo: %s\n", cudaGetErrorString(err));
            }

        }
        else{
            // =====================SGD UPDATE=========================
            // --- Update d_w_src and d_w_dst ---
            int num_blocks_w = (total_w + block_size - 1) / block_size;
            sgd_update_kernel<<<num_blocks_w, block_size>>>(d_w_src, d_grad_w_src, lr, total_w);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching sgd_update_kernel for d_w: %s\n", cudaGetErrorString(err));
            }

            sgd_update_kernel<<<num_blocks_w, block_size>>>(d_w_dst, d_grad_w_dst, lr, total_w);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching sgd_update_kernel for d_w: %s\n", cudaGetErrorString(err));
            }

            // --- Update attention vectors d_a ---
            int num_blocks_a = (total_a + block_size - 1) / block_size;
            sgd_update_kernel<<<num_blocks_a, block_size>>>(d_attn_vec, d_grad_attn_vec, lr, total_a);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching sgd_update_kernel for d_a: %s\n", cudaGetErrorString(err));
            }

            // --- Update output weights d_wo ---
            int total_wo = C * out_dim[L - 1];
            int num_blocks_wo = (total_wo + block_size - 1) / block_size;
            sgd_update_kernel<<<num_blocks_wo, block_size>>>(d_wo, d_grad_wo, lr, total_wo);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching sgd_update_kernel for d_wo: %s\n", cudaGetErrorString(err));
            }

        }
        //======================================================================

        cudaDeviceSynchronize();

       // After parameter update, reset gradients to zero
        cudaMemset(d_grad_w_src, 0, total_w * sizeof(float));
        cudaMemset(d_grad_w_dst, 0, total_w * sizeof(float));
        cudaMemset(d_grad_attn_vec, 0, total_a * sizeof(float));
        cudaMemset(d_grad_wo, 0, total_wo * sizeof(float));
        //memset the output gradients to zero
        for (int l = L-1; l >= 0; --l) {
            cudaMemset(input_gradients[l], 0, num_nodes * out_dim[l] * head[l] * sizeof(float));
        }

        // Stop timer
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        std::cout <<  " total time: " << elapsed.count() << " ms" << std::endl;

    }


}

