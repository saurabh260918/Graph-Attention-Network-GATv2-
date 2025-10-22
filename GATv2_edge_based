%%writefile cuda.cu
#include <cuda_runtime.h>
#include<cuda.h>
#include <curand_kernel.h>
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

// Compute softmax over a small array
__device__ void softmax(float* scores, int len) {       // scores is a pointer to an array of attenstion scores of length len of a perticular node.
    float max_val = scores[0];
    for (int i = 1; i < len; ++i) if (scores[i] > max_val) max_val = scores[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }
    for (int i = 0; i < len; ++i) scores[i] /= (sum+1e-8);
}




__global__ void reduce_sum_squares(float* grad, int n, float* out) {
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
    float* d_w,      // [num_layers][num_heads][out_dim][2*in_dim]
    float* d_a,      // [num_layers][num_heads][out_dim]
    float* d_wo,     // [C][out_dim_last_layer]
    const int*   head,     // [num_layers]
    const int*   in_dim,   // [num_layers]
    const int*   out_dim,  // [num_layers]
    int C,               // Number of output classes
    int    num_layers,
    curandState* states // Array of RNG states, length >= max needed threads
) {
    int l = blockIdx.x;
    int h = blockIdx.y;
    int o = threadIdx.x;

    // Compute global thread id
    int tid = l * gridDim.y * blockDim.x + h * blockDim.x + o;
    curandState local_state = states[tid];

    if (l < num_layers && h < head[l] && o < out_dim[l]) {
        int in_d = in_dim[l];
        int out_d = out_dim[l];
        float limit = sqrtf(6.0f / (2 * in_d + out_d));
        int w_offset = 0;
        for (int i = 0; i < l; ++i)
            w_offset += head[i] * out_dim[i] * 2 * in_dim[i];
        w_offset += h * out_dim[l] * 2 * in_dim[l];
        w_offset += o * 2 * in_dim[l];

        // Xavier: uniform(-limit, limit) for each weight
        for (int j = 0; j < 2 * in_d; ++j) {
            float rnd = curand_uniform(&local_state);   // in (0, 1]
            float value = rnd * 2.0f * limit - limit;   // in (-limit, limit)
            d_w[w_offset + j] = value;
        }
        // d_a vector: also uniform(-limit, limit)
        int a_offset = 0;
        for (int i = 0; i < l; ++i)
            a_offset += head[i] * out_dim[i];
        a_offset += h * out_dim[l];
        a_offset += o;
        float rnd = curand_uniform(&local_state);
        float value = rnd * 2.0f * limit - limit;
        d_a[a_offset] = value;
    }

    // Output layer weights
    if (l == num_layers - 1 && h == 0) {
        int out_d = out_dim[l];
        if (o < out_d) {
            float limit = sqrtf(6.0f / (C + out_d));
            for (int r = 0; r < C; ++r) {
                float rnd = curand_uniform(&local_state);
                float value = rnd * 2.0f * limit - limit;
                d_wo[r * out_d + o] = value;
            }
        }
    }

    // Save local state back to global memory
    states[tid] = local_state;

}

void clip_grad_norm(float* d_grad, int n, float clip_thresh) {
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

    float norm = sqrt(h_sumsq);
    float scale = 1.0f;
    if (norm > clip_thresh) {
        scale = clip_thresh / (norm + 1e-9f); // add small epsilon to avoid div by zero
    }

    // 3. apply scaling if needed
    if (scale < 1.0f) {
        scale_grads<<<blocks, threads>>>(d_grad, n, scale);
    }
}
__global__ void gatv2_edge_score_kernel(const float* __restrict__ input_features, const int* __restrict__ d_col_idx, const int* __restrict__ d_dst, const float* __restrict__ d_w, const float* __restrict__ d_a,  float* __restrict__ attn_score, int N, int in_dim, int out_dim, int H, int E, float negative_slope)  // H: num_heads, E: num_edges
 {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= H * E) return;

    int h = tid / E;   // head id
    int e = tid % E;   // edge id

    int src = d_col_idx[e];
    int dst = d_dst[e];

    const float* x_src = input_features + (size_t)src * in_dim;   //xi
    const float* x_dst = input_features + (size_t)dst * in_dim;   //xj  dst node features

    // Offsets into head-specific parameters and output slice
    const size_t W_head_stride = (size_t)out_dim * (2 * in_dim);
    const float* W_h = d_w + (size_t)h * W_head_stride;
    const float* a_h = d_a + (size_t)h * out_dim;
    float*       e_h = attn_score + (size_t)h * E;    //atten score for that head

    // e_ij = a_h^T ( LeakyReLU( W_h [h_i || h_j] ) )
    float e_val = 0.f;

    // Iterate over out_dim rows (one “channel” per row)
    for (int k = 0; k < out_dim; ++k) {
        const float* Wk_left = W_h + k * (2 * in_dim);

        // Dot with left half (x_src)
        float acc = 0.f;
        for (int d = 0; d < in_dim; ++d) {
            acc += Wk_left[d] * x_src[d];
        }

        // Dot with right half (x_dst)
        const float* Wk_right = Wk_left + in_dim;
        for (int d = 0; d < in_dim; ++d) {
            acc += Wk_right[d] * x_dst[d];
        }

        float s = leaky_relu(acc, negative_slope);
        e_val += a_h[k] * s;
    }

    // Write per-head, per-edge score in CSR edge order
    e_h[e] = e_val;
}

__global__ void compute_max_sum_attn_score(const int* __restrict__ d_row_ptr, const float* __restrict__ attn_score, int N, int H, int E, float* __restrict__ max_score, float* __restrict__ score_sum)
 {
    int dst = blockIdx.x;   // dst node id
    int h   = blockIdx.y;   // head id
    int lane = threadIdx.x; // 0..31

    int start = d_row_ptr[dst];
    int end   = d_row_ptr[dst + 1];

    // ---- Pass 1: Max ----
    float m = -1e9f;
    for (int e = start + lane; e < end; e += 32)
        m = fmaxf(m, attn_score[h * E + e]);

    // Warp reduce for max
    for (int offset = 16; offset > 0; offset /= 2)
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));

    m = __shfl_sync(0xffffffff, m, 0);  // broadcast

    // ---- Pass 2: Sum ----
    float s = 0.f;
    for (int e = start + lane; e < end; e += 32)
        s += __expf(attn_score[h * E + e] - m);

    // Warp reduce for sum
    for (int offset = 16; offset > 0; offset /= 2)
        s += __shfl_down_sync(0xffffffff, s, offset);

    if (lane == 0) {
        max_score[N * h + dst] = m;
        score_sum[N * h + dst] = s;
    }
}


__global__ void compute_attn_coeff(const int* __restrict__ d_col_idx, const int* __restrict__ d_dst, const float* __restrict__ attn_score, const float* __restrict__ d_max_attn_score, const float* __restrict__ d_sum_score_exp, float* __restrict__ attn_coeff, int E, int H, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E * H) return;

    int h = tid / E;   // head index
    int e = tid % E;   // edge index
    int dst = d_dst[e];

    // offset for max/sum arrays: each node has H heads
    int offset = dst + N * h; 

    float max_val = d_max_attn_score[offset];
    float sum_exp = d_sum_score_exp[offset];

    // softmax per edge
    float exp_score = __expf(attn_score[e + h * E] - max_val);  // numerically stable
    float alpha = exp_score / (sum_exp + 1e-8f);                // avoid divide by zero

    attn_coeff[e + h * E] = alpha;
    // if(tid == E*H -1)
    //     printf("compute_attn_coeff success\n");
}

__global__ void aggregate_kernel(
    const int* __restrict__ d_src,         // [E]
    const int* __restrict__ d_dst,         // [E]
    const float* __restrict__ d_attn_coeff,// [H * E] (head-major: h*E + e)
    const float* __restrict__ d_in_feat,   // [N * in_dim]
    const float* __restrict__ d_w,         // [H * out_dim * 2*in_dim]
    float* __restrict__ d_out_feat,        // [N * H * out_dim]    //output per head before activation
    int N, int H, int E, int in_dim, int out_dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= H * E) return;

    int h = tid / E;  // head id
    int e = tid % E;  // edge id

    int src = d_src[e];
    int dst = d_dst[e];

    // attention coefficient for this edge/head
    float alpha = d_attn_coeff[h * E + e];

    // head-specific weight matrix
    const float* w_h = d_w + h * (out_dim * 2 * in_dim);              // [out_dim, 2*in_dim]
    const float* x_src = d_in_feat + src * in_dim;              // [in_dim]
    float* out_dst_h = d_out_feat + ((dst * H + h) * out_dim);  // [out_dim]
    const float* w_h_left;


    // accumulate contribution into dst node’s output
    for (int k = 0; k < out_dim; ++k) {
        float sum = 0.0f;
        w_h_left = w_h + k * (2 * in_dim); // [2*in_dim] row k of W_h
        for (int j = 0; j < in_dim; ++j) {
            sum += w_h_left[j] * x_src[j];
        }
        sum *= alpha;
        atomicAdd(&out_dst_h[k], sum);
    }
}

__global__ void postActivationLayerOutput(
    const float* __restrict__ d_out_feat, // [N x H x out_dim]
    float* __restrict__ d_H,              // [N x H x out_dim] or [N x out_dim]
    int N, int H, int out_dim,
    bool is_last_layer,
    float negative_slope)
{
    int nodes_per_block = blockDim.x / out_dim; // e.g. 256/64 = 4
    int node_local = threadIdx.x / out_dim;     // which node inside the block
    int feat_idx   = threadIdx.x % out_dim;     // which feature index
    int node_id    = blockIdx.x * nodes_per_block + node_local;

    if (node_id >= N) return;

    if (is_last_layer) {
        // Average across heads
        float sum = 0.0f;
        for (int h = 0; h < H; h++) {
            size_t idx = ((size_t)node_id * H + h) * out_dim + feat_idx;
            float v = d_out_feat[idx];
            sum += (v > 0.0f ? v : v * negative_slope); // leaky relu
        }
        float avg = sum / H;
        d_H[node_id * out_dim + feat_idx] = avg;
    } else {
        // Store per head
        for (int h = 0; h < H; h++) {
            size_t idx = ((size_t)node_id * H + h) * out_dim + feat_idx;
            float v = d_out_feat[idx];
            v = (v > 0.0f ? v : v * negative_slope); // leaky relu
            d_H[idx] = v;
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

    extern __shared__ float s_wo[]; // Shared memory for W_o

    // Load W_o into shared memory
    int tid = threadIdx.x;
    int total_w = C * out_dim_last_layer;
    for (int i = tid; i < total_w; i += blockDim.x) {
        s_wo[i] = d_wo[i];
    }
    __syncthreads();
    // Thread/node index
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    // Pointer to this node's input feature (from previous layer)
    const float* node_output = &d_last_layer_output[node * out_dim_last_layer];

    // Pointer to this node's (node_id) output in d_z
    float* node_z = &d_z[node * C];

    // --- Compute matvec: z_i = W_o * input_i ---
    for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        const float* w_row = &s_wo[c * out_dim_last_layer];
        for (int j = 0; j < out_dim_last_layer; ++j) {
            acc += w_row[j] * node_output[j];
        }
        node_z[c] = acc;
    }

    // Softmax over channels for this node: softmax(z_i), result in-place at node_z
    softmax(node_z, C);

    // Copy results into d_y
    float* node_y = &d_y[node * C];
    for (int i = 0; i < C; ++i) {
        node_y[i] = node_z[i];
    }

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
    d_losses[idx] = -logf(fmaxf(prob, 1e-12f)); // cross-entropy loss

    // Find predicted class (argmax)
    float maxval = d_y[idx * C];
    int pred = 0;
    for (int c = 1; c < C; ++c) {
        float val = d_y[idx * C + c];
        if (val > maxval) { maxval = val; pred = c; }
    }
    d_corrects[idx] = (pred == label);
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
    // printf("\nLoss: %f, Accuracy: %.2f%%\n", loss, 100.0f * accuracy);
    return loss;
}


__global__ void compute_output_gradients(
    const float* d_y,        // [N][C], class probabilities after softmax
    const int* d_labels,     // [N],   true labels
    const float* d_hL,       // [N][out_dim_L], PRE-activation outputs
    const float* d_HL,       // [N][out_dim_L], POST-activation outputs
    const float* d_wo,       // [C][out_dim_L], output linear W
    float* grad_d_wo,        // [C][out_dim_L], output: grad for W_o
    float* grad_d_hL,        // [N][num_heads][out_dim_L], output: grad for h_i^L (pre-activation) per head
    int N, int C, int out_dim_L,
    int num_heads,
    float negative_slope     // LeakyReLU slope
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;
    extern __shared__ float shared[];     //size [blockDim.x][C]
    //Step 1: compute dL/dz = y_hat - y
    float* dL_dz = &shared[threadIdx.x * C];
    float* d_wo_shared = &shared[blockDim.x * C]; //size [C][out_dim_L]
    for (int c = 0; c < C; ++c) {
        dL_dz[c] = d_y[node*C + c] - (c == d_labels[node] ? 1.0f : 0.0f);
    }

    // Step 2: accumulate grad for W_o: dL/dWo += dL/dz * H_i_L^T
    for (int c = 0; c < C; ++c) {
        for (int d = 0; d < out_dim_L; ++d) {
            float contrib = (float)dL_dz[c] * (float)d_HL[node*out_dim_L + d];
            atomicAdd(&grad_d_wo[c*out_dim_L + d], contrib);
        }
    }

    // Step 3: backprop to H^(L): dL/dH^(L) = Wo^T * dL/dz, we are storing this in sum variable //// store the pre-activation gradient in grad_d_hL for that node. it will be 1/num_heads of the total gradient
    //fetch Wo to shared memory by all threads in the block
    for(int i= threadIdx.x; i < C*out_dim_L; i+= blockDim.x){
        d_wo_shared[i] = d_wo[i];
    }
    __syncthreads();
    //printf("\nDebug Info compute_output_gradients stage-1 \n");
    for (int d = 0; d < out_dim_L; ++d) {
        float sum = 0.0f;           //it is dL/dH after non-linearity
        for (int c = 0; c < C; ++c) {
            sum += d_wo_shared[c*out_dim_L + d] * dL_dz[c];
        }
      
        // Step 4: Chain rule through the non-linearity (LeakyReLU)
        float inv_heads = 1.0f / (float)num_heads;
        float h_val = d_hL[node*out_dim_L + d];
        float derivative = (h_val > 0.0f) ? 1.0f : negative_slope;       // H = f(h) = LeakyReLU(h)
        for (int h = 0; h < num_heads; ++h) {
            grad_d_hL[node * num_heads * out_dim_L + h * out_dim_L + d] =
                sum * derivative * inv_heads;
        }
    }
    // if(node == 0)
    //     printf("\ncompute_output_gradients success \n");
    
}


//.........................................................................................................//
__global__ void kernel_grad_atten_coeff(
    int num_edges,
    int num_heads,
    int in_dim,
    int out_dim,
    const int* __restrict__ src_idx,   // edge src indices (i)
    const int* __restrict__ dst_idx,   // edge dst indices (j)
    const float* __restrict__ features, // [num_nodes × in_dim]
    const float* __restrict__ d_w,    // [num_heads × out_dim × 2*in_dim]
    const float* __restrict__ grad_input, // [num_nodes × num_heads × out_dim]
    float* __restrict__ grad_attn_coeff,  // [num_heads × num_edges]
    bool is_last_layer
) {
    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_id >= num_edges) return;

    int i = src_idx[edge_id];  // source
    int j = dst_idx[edge_id];  // destination

    // loop over heads
    for (int h = 0; h < num_heads; ++h) {
        const float* W_head = d_w + h * out_dim * (2 * in_dim);  // W for head h
        const float* W_left = W_head;
        float tmp_vec = 0.0f;
        for (int d = 0; d < out_dim; ++d) {
            float Wxi_d = 0.0f;
            for (int k = 0; k < in_dim; ++k) {
                Wxi_d += W_left[d * (2 * in_dim) + k] * features[i * in_dim + k];
            }
            // dot with grad_input[i, h, :]
            float gj = grad_input[(j * num_heads + h) * out_dim + d];
            tmp_vec += gj * Wxi_d;
        }

        grad_attn_coeff[h * num_edges + edge_id] = tmp_vec;
        
    }
    // if(edge_id == num_edges-1)
    //     printf("\nkernel_grad_atten_coeff success\n");
}

// Kernel-2: compute grad wrt attention score e_{ij} for each (head, edge)
__global__ void compute_grad_attn_score_kernel(
    const int* __restrict__ d_row_ptr,     // [N+1]
    const int* __restrict__ d_dst,         // [E] -> dst node for each edge e
    const float* __restrict__ d_alpha,     // [H * E] head-major
    const float* __restrict__ d_grad_alpha,// [H * E] head-major
    float* __restrict__ d_grad_e,          // [H * E] output
    int N,
    int H,
    int E)                           // E may be large -> use 64-bit for some arithmetic
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * E;
    if (tid >= total) return;

    int h = (int)(tid / E);    // head id
    int e = (int)(tid % E);    // edge id

    int dst = d_dst[e];        // node j
    int start = d_row_ptr[dst];
    int end   = d_row_ptr[dst + 1];

    // load alpha_ij once
    const int base = h * E;
    float alpha_ij = d_alpha[base + e];

    float sum = 0.0f;

    // sum over k in N(j) -: k represented by edge index 'idx'
    for (int idx = start; idx < end; ++idx) {
        float alpha_kj = d_alpha[base + idx];         // alpha_{kj}
        float grad_alpha_kj = d_grad_alpha[base + idx];// dL/dalpha_{kj}

        // delta_{k,j} is 1 when idx == e i.e. k==j
        float delta = (idx == e) ? 1.0f : 0.0f;

        // contribution = grad_alpha_kj * alpha_kj * (delta - alpha_ij)
        sum += grad_alpha_kj * alpha_kj * (delta - alpha_ij);
    }

    d_grad_e[base + e] = sum;
    // if(tid == total -1)
    //     printf("compute_grad_attn_score_kernel success\n");
}

__global__ void compute_grad_parameters_kernel(
    int E,
    int H,
    int* __restrict__ d_src,               // [E]
    int* __restrict__ d_dst,               // [E]
    float* __restrict__ d_features,        // [N][in_dim]      //input features to this layer.
    float* __restrict__ d_input_gradients, // [N][H][out_dim]   //gradient flowing from upper layer to this layer. act as input gradient to this layer.
    float* __restrict__ d_grad_attn_score, // [H][E]
    float* __restrict__ d_attn_coeff,      // [H][E]
    float* __restrict__ d_w,               // [H][out_dim][2*in_dim]
    float* __restrict__ d_a,               // [H][out_dim]
    float* __restrict__ grad_w,            // [H][out_dim][2*in_dim]
    float* __restrict__ grad_a,            // [H][out_dim]
    int in_dim,
    int out_dim,
    float negative_slope
) {

    extern __shared__ float shmem[]; // size: 2*out_dim + 2*in_dim, first part: out_dim for attention vector, second part: out_dim for grad_a, third part: 2*in_dim for grad_w
    float* sh_a = &shmem[0];          // [size: out_dim]
    float* sh_grad_a = &shmem[out_dim]; // [size: out_dim]
    float* sh_grad_w = &shmem[2 * out_dim]; // [size: 2*in_dim]

    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int src = d_src[e];
    int dst = d_dst[e];
    
    // Loop over heads (outer loop)
    for (int h = 0; h < H; ++h) {
        const float* a_h = d_a + h * out_dim;                        // [out_dim]
        //fill sh_a from a_h using multiple threads
        for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
            sh_a[i] = a_h[i];
        }
        for (int i = threadIdx.x; i < out_dim; i += blockDim.x) {
            sh_grad_a[i] = 0.0f;
        }
        __syncthreads();
        const float* W_h = d_w + h * out_dim * (2 * in_dim);         // [out_dim][2*in_dim]
        float  dl_de = d_grad_attn_score[h * E + e];                 // ∂L/∂e_{ij}^{h}
        float  alpha_ij = d_attn_coeff[h * E + e];                   // α_{ij}^{h}
        const float* x_src = d_features + src * in_dim;    // x_i
        const float* x_dst = d_features + dst * in_dim;    // x_j
        const float* grad_dst = d_input_gradients + (dst * H + h) * out_dim; // ∂L/∂h_j^{h}
        float* grad_w_head = &grad_w[h * out_dim * (2 * in_dim)];         // [out_dim][2*in_dim]
        float* grad_a_head = &grad_a[h * out_dim];                        // [out_dim]

        for (int k = 0; k < out_dim; ++k) {
            for (int i = threadIdx.x; i < 2 * in_dim; i += blockDim.x) {
                sh_grad_w[i] = 0.0f;
            }
            __syncthreads();
            // compute s_k = W_left[k]·x_src + W_right[k]·x_dst
            const float* W_row = W_h + (size_t)k * (size_t)(2 * in_dim); // length 2*in_dim
            float s_k = 0.0f;
            // left part (src)
            for (int q = 0; q < in_dim; ++q) {
                s_k += W_row[q] * x_src[q];
            }
            // right part (dst)
            for (int q = 0; q < in_dim; ++q) {
                s_k += W_row[in_dim + q] * x_dst[q];
            }
            float leaky_s_k;
            if (s_k > 0)
                leaky_s_k = s_k;
            else
                leaky_s_k = s_k * negative_slope;
            // sh_grad_a[k] += dl_de * leakyRelu(s_k) 
            //but each thread is writing to same location. so use atomic add
            atomicAdd(&sh_grad_a[k], dl_de * leaky_s_k);
            float grad_dst_k = grad_dst[k]; // ∂L/∂h_j^{h}[k]
            for(int i=0; i<in_dim; i++){
                atomicAdd(&sh_grad_w[i], grad_dst_k * alpha_ij * x_src[i]);
            }
            float derivative = (s_k > 0.0f) ? 1.0f : negative_slope;
            float common_term = dl_de * sh_a[k] * derivative;
            
            for (int i = 0; i < in_dim; ++i) {
                atomicAdd(&sh_grad_w[i], common_term * x_src[i]);
            }
            for (int i = 0; i < in_dim; ++i) {
                atomicAdd(&sh_grad_w[in_dim + i], common_term * x_dst[i]);
            }
            __syncthreads();
            if(threadIdx.x==0){
                for (int i = 0; i < 2*in_dim; ++i) 
                    atomicAdd(&grad_w_head[k*2*in_dim + i], sh_grad_w[i]);
            }
            __syncthreads();

        }
        if(threadIdx.x==0){
            for(int i=0; i< out_dim; i++)
                atomicAdd(&grad_a_head[i], sh_grad_a[i]);
        }
        __syncthreads();
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
    const float* d_w,             // [H][out_dim][2*in_dim]
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
        int base = h * out_dim * 2 * in_dim;         // [out_dim][in_dim]
        float  dl_de = grad_attn_score[h * E + edge_id];                 // ∂L/∂e_{ij}^{h}
        float  alpha_ij = attn_coeff[h * E + edge_id];                   // α_{ij}^{h}
        const float* x_src = input_features + (src * in_dim);    // x_i
        const float* x_dst = input_features + (dst * in_dim);    // x_j
        const float* grad_dst = d_input_gradients + (dst * H + h) * out_dim; // ∂L/∂h_j^{h}
        float* grad_x_src = grad_x_features + (src * in_dim); // ∂L/∂x_i
        float* grad_x_dst = grad_x_features + (dst * in_dim); // ∂L/∂x_j
        // here each thread working on each edge will compute the contribution to input feature gradient of src and dst node
        // loop over output dimension
        const float* d_w_h = d_w + base;               // W_src part
        for (int od = 0; od < out_dim; ++od) {
            // Compute S_ij^{h}[od] = W_src[od,:] * xi + W_dst[od,:] * xj (on-the-fly)
            float sij_od = 0.0f;
            for (int id = 0; id < in_dim; ++id) {
                float w_src_od_id = d_w_h[od * 2 * in_dim + id];
                float w_dst_od_id = d_w_h[od * 2 * in_dim + id + in_dim];
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
                float w_src_od_id = d_w_h[od * 2 * in_dim + id];
                float w_dst_od_id = d_w_h[od * 2 * in_dim + in_dim + id];
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
    float* params, const float* grads, float* m, float* v,
    float lr, size_t n, float beta1, float beta2, float epsilon, int t) {
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


__global__ void sgd_update_kernel(float* params, float* grads, float lr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        params[i] -= lr * grads[i];
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
    //=======================================================================================//

    int max_degree = compute_max_degree(h_row_ptr, num_nodes);
    std::cout << "Max degree = " << max_degree << std::endl;

    auto itr = thrust::max_element(thrust::host, h_labels, h_labels + num_nodes);
    int C = *itr + 1; // Number of classes, assuming labels are 0-indexed
    std::cout << "Number of classes = " << C << std::endl;

    std::cout << "Graph loaded: " << num_nodes << " nodes, " << num_edges << " edges, "
              << "input_feature_vector_dim = " << input_dim << std::endl;



    int* in_dim = new int[L]; // Input dim for first layer, subsequent layers will be computed based on previous layer's output.
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
    float** d_layer_outputs = new float*[L]; // Output buffers per layer
    float** d_h = new float*[L];       // pre-nonlinearity output of hidden layers
    float* d_wo;       // Device linear transformation weight matrix of size C X out_dim.
    float* d_z;        // output after linear transformation. size is number of nodes X C.
    float* d_y;        // output probabilities [N][C]
    float** attn_score = new float*[L]; // Attention scores per layer [N][num_heads][max_degree]
    float** attn_coeff = new float*[L]; // Attention coefficients per layer [N][num_heads][max_degree]
    float** d_leakyrelu = new float*[L]; // LeakyReLU outputs per layer
    float** d_s = new float*[L];        // pre-nonlinearity output of each edge per layer
    float* grad_wo;    // [C][out_dim_last_layer]
    float* grad_d_w;   // [total_w]
    float* grad_d_a;   // [total_a]
    float negative_slope = 0.01f; // LeakyReLU slope

     // Start timer
    // auto start = std::chrono::high_resolution_clock::now();

     // 4. Initialize parameters and allocate memory
    cudaError_t err;
    // 1. Malloc and memcpy for graph data
    err = cudaMalloc(&d_features, num_nodes * in_dim[0] * sizeof(float));
    //printf("cudaMalloc d_features: %s\n", cudaGetErrorString(err));
    printf("\nsize of input features : %zu MB\n", (num_nodes * in_dim[0] * sizeof(float)) / (1024 * 1024));

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
    

    //==================CSR TO COO===================
    int* d_src;
    int* d_dst;
    cudaMalloc(&d_src, num_edges * sizeof(int));
    cudaMalloc(&d_dst, num_edges * sizeof(int));
    printf("\nsize of graph data (CSR,COO, labels) : %zu KB\n", (((num_nodes+1) + num_edges + num_nodes + 2* num_edges)* sizeof(int)) / (1024));

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    csr_to_coo_kernel<<<blocks, threads>>>(d_row_ptr, d_col_idx, d_src, d_dst, num_nodes);
    cudaDeviceSynchronize();
    //check for error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching csr_to_coo_kernel: %s\n", cudaGetErrorString(err));
    }
            

    //==================END============================
    size_t total_size = 0;
    for (int l = 0; l < L; ++l) {
        size_t output_size;
        if (l == L - 1) {
            // Last layer: average heads, only out_dim[l] per node
            output_size = num_nodes * out_dim[l];
        } else {
            // Non-last layer: concatenate heads
            output_size = num_nodes * head[l] * out_dim[l];
        }
        total_size += output_size;
        cudaMalloc(&d_layer_outputs[l], output_size * sizeof(float));
        cudaMalloc(&d_h[l], output_size * sizeof(float));
    }
    printf("\nTotal size of layer outputs(pre & post activation): %zu MB\n", (2*total_size * sizeof(float)) / (1024 * 1024));
    int total_heads = 0;
    for (int l = 0; l < L; ++l) {
        total_heads += head[l];
        cudaMalloc(&attn_score[l], head[l] * num_edges * sizeof(float));
        cudaMalloc(&attn_coeff[l], head[l] * num_edges * sizeof(float));
        //cudaMalloc(&d_leakyrelu[l], num_nodes * head[l] * max_degree * out_dim[l] * sizeof(float));
        //cudaMalloc(&d_s[l], num_nodes * head[l] * max_degree * out_dim[l] * sizeof(float));
    }
    printf("Total size of attention scores & coeffs: %zu MB\n", ((2*total_heads* num_edges * sizeof(float)) / (1024 * 1024)));

    //==========================////////////============================================================

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
    int* w_offset = new int[L];
    int* a_offset = new int[L];
    w_offset[0]= 0; // Offsets for weights
    a_offset[0]= 0; // Offsets for attention vectors
    for (int l = 1; l < L; ++l) {
        w_offset[l] = w_offset[l-1] + head[l-1] * out_dim[l-1] * 2 * in_dim[l-1];
        a_offset[l] = a_offset[l-1] + head[l-1] * out_dim[l-1];
        total_w += head[l-1] * out_dim[l-1] * 2 * in_dim[l-1];
        total_a += head[l-1] * out_dim[l-1];
    }
    total_w += head[L-1] * out_dim[L-1] * 2 * in_dim[L-1]; // Last layer weights
    total_a += head[L-1] * out_dim[L-1]; // Last layer attention vectors

    cudaMalloc(&d_w, total_w * sizeof(float));
    cudaMalloc(&d_a, total_a * sizeof(float));
    cudaMalloc(&d_wo, C * out_dim[L - 1] * sizeof(float));
    cudaMalloc(&d_z, num_nodes * C * sizeof(float));
    cudaMalloc(&d_y, num_nodes * C * sizeof(float));
    cudaMalloc(&grad_wo, C * out_dim[L-1] * sizeof(float));
    cudaMemset(grad_wo, 0, C * out_dim[L-1] * sizeof(float)); // Initialize to zero
    cudaMalloc(&grad_d_w, total_w * sizeof(float));
    cudaMemset(grad_d_w, 0, total_w * sizeof(float)); // Initialize to zero
    cudaMalloc(&grad_d_a, total_a * sizeof(float));
    cudaMemset(grad_d_a, 0, total_a * sizeof(float)); // Initialize to zero

    //print parameters and their gradient storage size cumulatively
    printf("\nTotal size of parameters and their gradients: %zu MB\n", ( ((total_w + total_a + (C*out_dim[L-1])) * 2 * sizeof(float)) / (1024 * 1024)));

    int max_heads = *std::max_element(head, head + L);  // Maximum number of heads across all layers
    int max_out_dim = *std::max_element(out_dim, out_dim + L); // Maximum output dimension across all layers

    //=============INITIALISE ADAM PARAMETERS========================
    float *m_w;
    cudaMalloc(&m_w, total_w * sizeof(float));
    cudaMemset(m_w, 0, total_w * sizeof(float));
    float *v_w;
    cudaMalloc(&v_w, total_w * sizeof(float));
    cudaMemset(v_w, 0, total_w * sizeof(float));

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

    //===============INITIALISE XAVIER WEIGHTS================================================================
    // 3. Initialize weights and attention vectors using Xavier initialization

    int nthreads = L * max_heads * max_out_dim;
    curandState* d_states;
    cudaMalloc(&d_states, nthreads * sizeof(curandState));
    int blockSize = 128;
    dim3 gridSize((nthreads + blockSize - 1) / blockSize);
    setup_states_kernel<<<gridSize, blockSize>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching setup_states_kernel: %s\n", cudaGetErrorString(err));
    }

    // 2. Call Xavier initialization kernel
    dim3 grid(L, max_heads);
    int block = max_out_dim;
    xavier_init_kernel_curand<<<grid, block>>> (d_w, d_a, d_wo, d_head, d_in_dim, d_out_dim, C, L, d_states);
    cudaDeviceSynchronize();

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching xavier_init_kernel: %s\n", cudaGetErrorString(err));
    }

    

    float* d_loss;      // [num_nodes]
    int*   d_correct;   // [num_nodes]
    cudaMalloc(&d_loss,    num_nodes * sizeof(float));
    cudaMalloc(&d_correct, num_nodes * sizeof(int));
    printf("\nloss and accuracy storage: %zu MB\n", ((num_nodes * (sizeof(float)+ sizeof(int))) / (1024 * 1024)));

    //======================INTERMEDIATE GRADIENTS BUFFERS===========================
    
    // // Allocate input gradient buffers: one per layer.
    
    size_t total_input_grad_size = 0;

    float** input_gradients = new float*[L];     // Array of pointers to output gradients per layer//these are the gradients which are coming out of that layer which will become as input gradient to previous layer
    for (int l = L-1; l >= 0; --l) {
        size_t input_grad_size = (size_t)num_nodes * head[l] * out_dim[l] * sizeof(float);
        total_input_grad_size += input_grad_size;
        cudaMalloc(&input_gradients[l], input_grad_size);
        cudaMemset(input_gradients[l], 0, input_grad_size); // Initialize to zero
    }
    float* grad_attn_score; // Gradient of attention scores for edges  // these we will utilise at each layer  during backprop.
    float* grad_attn_coeff; // Gradient of attention coefficients for edges
    cudaMalloc(&grad_attn_score, max_heads * num_edges * sizeof(float));
    cudaMalloc(&grad_attn_coeff, max_heads * num_edges * sizeof(float));
    printf("Total size of intermediate gradients: %zu MB\n", ((total_input_grad_size + max_heads * num_edges * 2) * sizeof(float)) / (1024 * 1024));
    //===============================================================================
    
    size_t max_size = num_nodes * max_heads * sizeof(float);
    float* d_max_attn_score;
    float* d_sum_score_exp;
    cudaMalloc(&d_max_attn_score, max_size);
    cudaMalloc(&d_sum_score_exp, max_size);

    //**************************||**************************************//
    // Measure GPU memory after all allocations
    size_t free_after;
    cudaMemGetInfo(&free_after, &total_mem);
    double used_mb = (double)(free_before - free_after) / (1024.0 * 1024.0);

    printf("\n[Memory Tracker] After all allocations:\n");
    printf("  Free GPU memory : %.2f MB\n", free_after / (1024.0 * 1024.0));
    printf("  Approx. GPU memory allocated by this program: %.2f MB\n", used_mb);
    //**************************||**************************************//
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        printf("\nEpoch %d\n", epoch);
        // b. Forward pass for each layer
        float* d_layer_inputs = d_features;
        for (int l = 0; l < L; ++l) {
            // printf("\nLayer %d forward pass:\n", l);
            const float* d_w_l = d_w + w_offset[l];
            const float* d_a_l = d_a + a_offset[l];
        

            //=======================================FORWARD PASS START HERE============================
            // Step 1: Compute attention scores for all edges and heads
            int threads_per_block = 256;
            long long total_threads = (long long)head[l] * num_edges;                    // may be 32-bit overflow
            int total_blocks  = (int)((total_threads + threads_per_block - 1) / threads_per_block);
            gatv2_edge_score_kernel<<<total_blocks, threads_per_block>>>(d_layer_inputs,  d_col_idx,  d_dst, d_w_l, d_a_l, attn_score[l], num_nodes, in_dim[l], out_dim[l], head[l], num_edges, negative_slope);
            cudaDeviceSynchronize();
            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching gatv2_edge_score_kernel: %s\n", cudaGetErrorString(err));
            } 
            
            dim3 grid(num_nodes, head[l]);     // one block per (dst, h)
            dim3 block(32);      // one warp
            
            // Step 2: Compute max attention scores and sum of exp for attention scores per node and head
            compute_max_sum_attn_score<<<grid, block>>>(d_row_ptr, attn_score[l], num_nodes, head[l], num_edges, d_max_attn_score, d_sum_score_exp);
            cudaDeviceSynchronize();
            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching compute_max_sum_attn_score: %s\n", cudaGetErrorString(err));
            }

            // Step 3: Compute attention coefficients for all edges and heads
            compute_attn_coeff<<<total_blocks, threads_per_block>>>(d_col_idx, d_dst, attn_score[l], d_max_attn_score, d_sum_score_exp, attn_coeff[l], num_edges, head[l], num_nodes);  
            cudaDeviceSynchronize();
            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching compute_attn_coeff: %s\n", cudaGetErrorString(err));
            }

            // Step 4: Aggregate features for each dst node and head
            aggregate_kernel<<<total_blocks, threads_per_block>>>(d_src, d_dst, attn_coeff[l], d_layer_inputs, d_w_l, d_h[l], num_nodes, head[l], num_edges, in_dim[l], out_dim[l]);
            cudaDeviceSynchronize();
            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching aggregate_kernel: %s\n", cudaGetErrorString(err));
            }

            bool is_last_layer = (l == L - 1);
            int num_threads = 256;
            int nodes_per_block = num_threads / out_dim[l];
            int grid_size = (num_nodes + nodes_per_block - 1) / nodes_per_block;
            postActivationLayerOutput<<<grid_size, num_threads>>>(d_h[l], d_layer_outputs[l], num_nodes, head[l], out_dim[l], is_last_layer, 0.01f);
            cudaDeviceSynchronize();
            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching postprocess_layer: %s\n", cudaGetErrorString(err));
            }
            d_layer_inputs = d_layer_outputs[l];


        }
        
    
        
        int threads_out = 128;
        int num_blocks_out = (num_nodes + threads_out - 1) / threads_out;
        size_t shared_memory = (C) * out_dim[L - 1] * sizeof(float);
        // Step 5: Final linear transformation to get class scores
        gatv2_output_kernel<<<num_blocks_out, threads_out, shared_memory>>>(d_wo,  d_layer_inputs, d_z, d_y, num_nodes, C, out_dim[L - 1]);
        cudaDeviceSynchronize();
        // Check for errors after gatv2_output_kernel
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error launching gatv2_output_kernel: %s\n", cudaGetErrorString(err));
        }
       
        // c. ------------------------Calculate loss and accuracy -------------------------------------
        int threads_per_block_acc = 256;
        int num_blocks_acc = (num_nodes + threads_per_block_acc - 1) / threads_per_block_acc;
        compute_loss_accuracy_kernel<<<num_blocks_acc, threads_per_block_acc>>>(d_y, d_labels, d_loss, d_correct, num_nodes, C);
        cudaDeviceSynchronize();

        compute_loss_and_accuracy(num_nodes, d_loss, d_correct);
        //---------------------------------------------------------------------------------------------

        // d. Backward pass
        // 1: Output layer gradient
        int threads_per_block_outGrad = 128;
        int num_blocks = (num_nodes + threads_per_block_outGrad - 1) / threads_per_block_outGrad;
        size_t shared_mem = ((C * threads_per_block_outGrad) + (C*out_dim[L-1])) * sizeof(float);
        compute_output_gradients<<<num_blocks, threads_per_block_outGrad, shared_mem>>>(
            d_y, d_labels, d_h[L-1], d_layer_outputs[L-1], d_wo, grad_wo,
            input_gradients[L-1], num_nodes, C, out_dim[L-1], head[L-1], negative_slope
        );
        cudaDeviceSynchronize();
        // Check for errors after compute_output_gradients
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error launching compute_output_gradients: %s\n", cudaGetErrorString(err));
        }

        

        // 2: Loop backward through GAT layers
        float* grad_input = input_gradients[L-1]; // Start with last layer's input gradient buffer
        for (int l = L-1; l >= 0; --l) {
            //printf("\nBackward pass for layer %d\n", l);
            int threads_grad_alpha = 256;
            int blocks_grad_alpha = (num_edges + threads_grad_alpha - 1) / threads_grad_alpha;
            float* d_w_l = d_w + w_offset[l];
            bool is_last_layer = (l == L - 1);
            kernel_grad_atten_coeff<<<blocks_grad_alpha, threads_grad_alpha>>>(
                num_edges, head[l], in_dim[l], out_dim[l], d_src, d_dst, (l > 0) ? d_layer_outputs[l - 1] : d_features, 
                d_w_l, grad_input, grad_attn_coeff, is_last_layer);
            cudaDeviceSynchronize();
            // Check for errors after kernel_grad_atten_coeff
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching kernel_grad_atten_coeff: %s\n", cudaGetErrorString(err));
            }

            int threads_grad_score = 256;
            int total_threads = head[l] * num_edges;
            int blocks_grad_score = (int)((total_threads + threads_grad_score - 1) / threads_grad_score);

            compute_grad_attn_score_kernel<<<blocks_grad_score, threads_grad_score>>>(
                d_row_ptr, d_dst, attn_coeff[l], grad_attn_coeff, grad_attn_score,
                num_nodes, head[l], num_edges
            );
            cudaDeviceSynchronize();
            // Check for errors after compute_grad_attn_score_kernel
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching compute_grad_attn_score_kernel: %s\n", cudaGetErrorString(err));
            }

            int threads_grad_a_w = 256;
            int block_grad_a_w = (num_edges + threads_grad_a_w - 1) / threads_grad_a_w;
            float shared_mem_a_w = 2* (in_dim[l] + out_dim[l]) * sizeof(float);
            compute_grad_parameters_kernel<<<block_grad_a_w, threads_grad_a_w, shared_mem_a_w>>>(
                num_edges, head[l], d_src, d_dst,
                (l > 0) ? d_layer_outputs[l - 1] : d_features, grad_input, grad_attn_score, attn_coeff[l], d_w + w_offset[l], d_a + a_offset[l],
                grad_d_w + w_offset[l], grad_d_a + a_offset[l], in_dim[l], out_dim[l], negative_slope
            );
            cudaDeviceSynchronize();
            // Check for errors after compute_grad_parameters_kernel
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching compute_grad_parameters_kernel: %s\n", cudaGetErrorString(err));
            }
            if(l == 0) break;

            int threads_grad_input = 256;
            int blocks_grad_input = (num_edges + threads_grad_input - 1) / threads_grad_input;
            float shared_mem_input = out_dim[l] * sizeof(float);
            compute_features_input_gradients<<<blocks_grad_input, threads_grad_input, shared_mem_input>>>(
                num_nodes, head[l], num_edges, in_dim[l], out_dim[l],
                negative_slope, d_src, d_dst, attn_coeff[l], d_layer_outputs[l-1], d_w + w_offset[l],
                grad_input, grad_attn_score, d_a + a_offset[l], input_gradients[l-1]
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
            // Prepare input gradient for next iteration down the stack
            grad_input = input_gradients[l-1]; // Output gradient of current layer becomes input gradient for previous layer
        }


        // -----> *** 5. PARAMETER UPDATE SECTION —  ***
        if(clip){
            //---------_CLIP GRADIENTS___________________
            clip_grad_norm(grad_d_w, total_w, 5.0f);   //  threshold=5
            clip_grad_norm(grad_d_a, total_a, 5.0f);
            clip_grad_norm(grad_wo, C * out_dim[L - 1], 5.0f);
            // //_________________________________________________
        }

        int block_size = 256;
        if(optimizer == "adam") {
            //================ADAM UPDATE=========================

            //--- Update d_w ---
            int num_blocks_w = (total_w + block_size - 1) / block_size;
            adam_update_kernel<<<num_blocks_w, block_size>>>(d_w, grad_d_w, m_w, v_w, lr, total_w, beta1, beta2, 1e-8f, epoch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching adam_update_kernel for d_w: %s\n", cudaGetErrorString(err));
            }

            // --- Update attention vectors d_a ---
            int num_blocks_a = (total_a + block_size - 1) / block_size;
            adam_update_kernel<<<num_blocks_a, block_size>>>(d_a, grad_d_a, m_a, v_a, lr, total_a, beta1, beta2, 1e-8f, epoch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching adam_update_kernel for d_a: %s\n", cudaGetErrorString(err));
            }

            // --- Update output weights d_wo ---
            int total_wo = C * out_dim[L - 1];
            int num_blocks_wo = (total_wo + block_size - 1) / block_size;
            adam_update_kernel<<<num_blocks_wo, block_size>>>(d_wo, grad_wo, m_wo, v_wo, lr, total_wo, beta1, beta2, 1e-8f, epoch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching adam_update_kernel for d_wo: %s\n", cudaGetErrorString(err));
            }
        }
        else if(optimizer == "sgd")
        {

            // -------SGD UPDATE-----------------
            int num_blocks_w = (total_w + block_size - 1) / block_size;
            sgd_update_kernel<<<num_blocks_w, block_size>>>(d_w, grad_d_w, lr, total_w);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching sgd_update_kernel for d_w: %s\n", cudaGetErrorString(err));
            }

            // --- Update attention vectors d_a ---
            int num_blocks_a = (total_a + block_size - 1) / block_size;
            sgd_update_kernel<<<num_blocks_a, block_size>>>(d_a, grad_d_a, lr, total_a);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching sgd_update_kernel for d_a: %s\n", cudaGetErrorString(err));
            }

            // --- Update output weights d_wo ---
            int total_wo = C * out_dim[L - 1];
            int num_blocks_wo = (total_wo + block_size - 1) / block_size;
            sgd_update_kernel<<<num_blocks_wo, block_size>>>(d_wo, grad_wo, lr, total_wo);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching sgd_update_kernel for d_wo: %s\n", cudaGetErrorString(err));
            }
        }

        cudaDeviceSynchronize();

        // 6. Reset gradients to zero for next epoch

        cudaMemset(grad_d_w, 0, total_w * sizeof(float));
        cudaMemset(grad_d_a, 0, total_a * sizeof(float));
        cudaMemset(grad_wo, 0, total_wo * sizeof(float));
        //memset the output gradients to zero
        for (int l = 0; l < L; ++l) {
            cudaMemset(input_gradients[l], 0, num_nodes * out_dim[l] * head[l] * sizeof(float));
        }
        // Stop timer
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        std::cout <<  " total time: " << elapsed.count() << " ms" << std::endl;
    }



}

