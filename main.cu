//%%writefile cuda.cu
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

//includes float64 for double precision using atomicCAS for parameters grad compute

// including gradient_clip

double compute_gradient_norm(const double* grad_array, size_t size) {
    double sum_squares = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum_squares += grad_array[i] * grad_array[i];
    }
    return sqrt(sum_squares);
}


__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
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
    // float avg_loss = total_loss / N;
    float loss = total_loss;
    float accuracy = static_cast<float>(total_correct) / N;
    // printf("\nAvg Loss: %f, Accuracy: %.2f%%\n", avg_loss, 100.0f * accuracy);
    printf("\nLoss: %f, Accuracy: %.2f%%\n", loss, 100.0f * accuracy);
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




__global__ void reduce_sum_squares(const double* grad, int n, double* out) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i < n) {
        double g = grad[i];
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



__global__ void scale_grads(double* grad, int n, double scale) {
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

void clip_grad_norm(double* d_grad, int n, double clip_thresh) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // allocate device scalar for sum of squares
    double zero = 0.0;
    double* d_sumsq;
    cudaMalloc(&d_sumsq, sizeof(double));
    cudaMemcpy(d_sumsq, &zero, sizeof(double), cudaMemcpyHostToDevice);

    // 1. compute total sum of squares
    reduce_sum_squares<<<blocks, threads, threads*sizeof(double)>>>(d_grad, n, d_sumsq);

    // 2. copy result back to host (just one double)
    double h_sumsq;
    cudaMemcpy(&h_sumsq, d_sumsq, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sumsq);

    double norm = sqrt(h_sumsq);
    double scale = 1.0;
    if (norm > clip_thresh) {
        scale = clip_thresh / (norm + 1e-9);
    }

    // 3. apply scaling if needed
    if (scale < 1.0) {
        scale_grads<<<blocks, threads>>>(d_grad, n, scale);
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
    //if thread 0 node 0 then print address of d_features
    // if (node == 0 && head == 0 && threadIdx.x == 0) {
    //     printf("Address of d_features recieved by kerenl: %p\n", (void*)d_features);
    // }

    // if (node == 0 && head == 0 && threadIdx.x == 0) {
    //     printf("\nFeatures of node 0: ");
    //     for (int i = 0; i < in_dim; ++i) {
    //         printf("%f ", d_features[node * in_dim + i]);
    //     }
    //     printf("\n");
    // }
    if (node >= N || head >= num_heads || threadIdx.x >= max_degree) return;

    extern __shared__ float shared_mem[];
    float* attn_scores = shared_mem;         // [max_degree]
    float* head_output = &shared_mem[max_degree]; // [out_dim]
    int row_start = d_row_ptr[node];
    int row_end = d_row_ptr[node + 1];
    int degree = row_end - row_start;

    for (int i = threadIdx.x; i < out_dim; i += blockDim.x)
        head_output[i] = 0.0f;

    if (threadIdx.x >= degree) return;

    int j = d_col_idx[row_start + threadIdx.x];


    // compute s_i_j i.e. dot product of W_row and [x_i || x_j] without forming concat_x
    int base_s = ((((node * num_heads + head) * max_degree) + threadIdx.x) * out_dim);
    for (int od = 0; od < out_dim; ++od) {
        float acc = 0.0f;    //accumulation
        // left half uses x_i
        const float* W_left = &d_W[(head * out_dim * 2 * in_dim) + (od * 2 * in_dim) + 0];
        for (int id = 0; id < in_dim; ++id) {
            acc += W_left[id] * d_features[node * in_dim + id];
        }
        // right half uses x_j
        const float* W_right = &d_W[(head * out_dim * 2 * in_dim) + (od * 2 * in_dim) + in_dim];
        int j_idx = j * in_dim;
        for (int id = 0; id < in_dim; ++id) {
            acc += W_right[id] * d_features[j_idx + id];
        }
        // now acc == dot(W_row, concat(x_i, x_j))
        d_s[base_s + od] = acc;
    }


    // LeakyReLU & store per-edge per-dim
    //for (int k = 0; k < out_dim; ++k) s[k] = leaky_relu(s[k]);
    for(int k = 0; k < out_dim; ++k) {
        d_leakyrelu[base_s + k] = leaky_relu(d_s[base_s + k]);  // Store leaky_relu(s_ij) per neighbor/edge
    }

    // Compute attention score: e_ij = a^T s (after nonlinearity)
    const float* a_head = &d_a[head * out_dim];
    float attnscore_val = dot(a_head, &d_leakyrelu[base_s], out_dim);
    //printf("attention score: %f\n", attnscore_val);
    attn_scores[threadIdx.x] = attnscore_val;
    attn_score[(node * num_heads + head) * max_degree + threadIdx.x] = attnscore_val;
    __syncthreads();

    if (threadIdx.x == 0) softmax(attn_scores, degree);
    //store softmax(attn_scores) in attn_coeff by each thread since total active threads < max_degree
    attn_coeff[(node * num_heads + head) * max_degree + threadIdx.x] = attn_scores[threadIdx.x];
    __syncthreads();


    int base = head * out_dim * 2 * in_dim;
    for (int od = 0; od < out_dim; ++od) {
        float sum = 0.0f;
        for (int id = 0; id < in_dim; ++id) {
            float w_val = d_W[base + od * 2 * in_dim + in_dim + id];
            float x_val = d_features[j * in_dim + id];
            sum += w_val * x_val;
        }
        atomicAdd(&head_output[od], attn_scores[threadIdx.x] * sum);
    }

    __syncthreads();

    if (!is_last_layer) {
        // add Nonlinearity then concatenate
        for (int i = threadIdx.x; i < out_dim; i += degree) {
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
            for (int i = 0; i < out_dim; ++i)
                d_h[node * out_dim + i] = d_out[node * out_dim + i];
        }
        __syncthreads();
        // Nonlinearity after averaging
        if (head == 0) {
            for (int i = threadIdx.x; i < out_dim; i += degree) {
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
    // if (node == 0) {
    //     printf("d_y for node 0: ");
    //     float sum=0;
    //     for (int k = 0; k < C; ++k) {
    //         printf("%f ", d_y[k]);
    //         sum += d_y[k];
    //     }
    //     printf("\n");
    //     printf("Sum of probabilities for node 0: %f\n", sum);
    // }
}

__global__ void compute_output_gradients(
    const float* d_y,        // [N][C], softmax output
    const int* d_labels,     // [N],   true labels
    const float* d_hL,       // [N][out_dim_L], PRE-nonlinearity (input to activation)
    const float* d_HL,       // [N][out_dim_L], POST-nonlinearity (output of activation)
    const float* d_wo,       // [C][out_dim_L], output linear W
    double* grad_d_wo,        // [C][out_dim_L], output: grad for W_o
    float* grad_d_hL,        // [N][num_heads][out_dim_L], output: grad for h_i^L (pre-activation) per head
    int N, int C, int out_dim_L,
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

    // // Print dL_dz of size N x C
    // printf("\ndL_dz values: \n");
    // printf("Node %d: ", node);
    // for (int c = 0; c < C; ++c) {
    //     printf("%.17f ", dL_dz[c]);
    // }
    // printf("\n");

    // Step 2: accumulate grad for W_o: dL/dWo += dL/dz * H_i_L^T
    for (int c = 0; c < C; ++c) {
        for (int d = 0; d < out_dim_L; ++d) {
            double contrib = (double)dL_dz[c] * (double)d_HL[node*out_dim_L + d];
            atomicAddDouble(&grad_d_wo[c*out_dim_L + d], contrib);
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
    // if (node == 0) {
    //     printf("grad_d_hL for node 0: ");
    //     for (int i = 0; i < 20 && i < num_heads * out_dim_L; ++i) {
    //         printf("%f ", grad_d_hL[i]);
    //     }
    //     printf("\n");
    // }
    if (node == 0) {
        //printf("Size of grad_d_hL for node 0: %d\n", num_heads * out_dim_L);
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
    double* grad_w,            // [num_heads][out_dim][2*in_dim]
    double* grad_a,            // [num_heads][out_dim]
    float* grad_x_lower,      // [N][in_dim]    // gradient of input features
    float negative_slope,
    int max_degree,
    bool last_layer, // 0 if not last layer, 1 if last layer
    int layer,
    unsigned long long total_grad_w_size
) {
    int i = blockIdx.x;    // Node index
    int h = blockIdx.y;    // Head index
    int tid = threadIdx.x; // Neighbor offset
    //create a shared memory equal to max_degree+out_dim
    extern __shared__ float shared_memory[];    //size max_degree
    float* dL_d_h = &shared_memory[max_degree]; // size out_dim
    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-1 \n");
    }
    // if(layer==1 && i==0 && h==0 && tid==0){
    //     printf("grad_w size: %d\n", num_heads * out_dim * 2 * in_dim    );
    // }

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
    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-2 \n");
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
    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-3 \n");
    }

    // Store dL/d alpha_ij in shared memory for later use
    shared_memory[tid] = dL_d_alpha_ij;
    __syncthreads(); // Ensure all threads have written their values

    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-3i \n");
    }

    // --- Step D.4: grad_W direct ---
    float alpha = attn_coeff[(i * num_heads + h) * max_degree + (tid)];

    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-3ii \n");
    }

    for (int od = 0; od < out_dim; ++od) {
        for (int id = 0; id < in_dim; ++id) {
            double contrib = (double)alpha * (double)dL_d_h[od] * (double)d_x[j*in_dim + id];
            atomicAddDouble(&grad_w[(h*out_dim*2*in_dim) + (od*2*in_dim) + (in_dim + id)], contrib);
        }
    }



    // --- Step E.1: dL/d e_ij ---
    float dL_d_e_ij = 0.0f;
    float alpha_ij = attn_coeff[(i * num_heads + h) * max_degree + tid];
    // for (int kk = 0; kk < deg; ++kk) {
    //     int k = d_col_idx[row_start + kk];
    //     float alpha_ik = attn_coeff[(i * num_heads + h) * max_degree + kk];
    //     // For dL_d_alpha_ik, may require shared or global memory,
    //     // or recompute in another pass for full parallel safety
    //     float dL_d_alpha_ik = shared_memory[kk];
    //     dL_d_e_ij += dL_d_alpha_ik * alpha_ik * ((j == k ? 1.0f : 0.0f) - alpha_ij);
    // }
    for (int kk = 0; kk < deg; ++kk) {
        float alpha_ik = attn_coeff[(i * num_heads + h) * max_degree + kk];
        float dL_d_alpha_ik = shared_memory[kk];

        if (kk == tid) {
            // ∂α_ik/∂e_ij when k=j: α_ij * (1 - α_ij)
            dL_d_e_ij += dL_d_alpha_ik * alpha_ij * (1.0f - alpha_ij);
        } else {
            // ∂α_ik/∂e_ij when k≠j: -α_ik * α_ij
            dL_d_e_ij += dL_d_alpha_ik * (-alpha_ik * alpha_ij);
        }
    }



    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-4i \n");
    }

    // --- Step E.2: grad_a ---
    int leaky_base = (((i * num_heads + h) * max_degree) + tid) * out_dim;
    for (int od = 0; od < out_dim; ++od) {
        double contrib = (double)dL_d_e_ij * (double)d_leakyrelu[leaky_base + od];
        atomicAddDouble(&grad_a[h * out_dim + od], contrib);
    }

    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-4ii \n");
    }

    // --- Step E.3: grad_W via attention ---
    float* s_ij = &d_s[leaky_base]; // each [out_dim]
    //float leaky_grad_val;
    for (int od = 0; od < out_dim; ++od) {
        float leaky_grad_val = (s_ij[od] > 0) ? 1.0f : negative_slope;
        double elem = (double)d_a[h * out_dim + od] * (double)leaky_grad_val * (double)dL_d_e_ij;
        for (int id = 0; id < 2 * in_dim; ++id) {
            float x_concat = (id < in_dim) ? d_x[i*in_dim + id] : d_x[j*in_dim + (id-in_dim)];
            double contrib = elem * (double)x_concat;
            atomicAddDouble(&grad_w[h * out_dim * 2 * in_dim + od * 2 * in_dim + id], contrib);
        }
    }

    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-5 \n");
    }

    // ---- DIRECT GRADIENT FOR x_i ----
    // For node i as a neighbor of node j, accumulate to grad_x_lower[i][*] as per direct formula
    int offset = -1;   //(will create problem if directed edge)
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

    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-6 \n");
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
    if (i == 0 && tid==0) {
        //printf("Debug Info gatv2_layer_backward stage-last \n");
    }

}


__global__ void sgd_update_kernel(float* params, const double* grads, float lr, size_t n) {
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




int main() {
    // 1. Load graph in CSR format
    int num_nodes, num_edges, input_dim;
    float* h_features;    // [num_nodes][input_dim]
    int* h_row_ptr;       // [num_nodes+1]
    int* h_col_idx;       // [num_edges]
    int* h_labels;       // [num_nodes]
    //std::cout << "Loaded labels length = 200" << "\n";
    //std::cout << "Expected num_nodes = 200"  << "\n";
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
    const int L = 2; // Example: 5 layers
    int C = 7; // C class classification problem.
    int head[L] = {4,1};         // Number of heads per layer
    int out_dim[L] = {150, 7};    // Output dim per head per layer
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
    float* attn_score[L]; // Attention scores per layer [N][num_heads][max_degree]
    float* attn_coeff[L]; // Attention coefficients per layer [N][num_heads][max_degree]
    float* d_leakyrelu[L]; // LeakyReLU outputs per layer
    float* d_s[L];        // pre-nonlinearity output of each edge per layer
    double* grad_wo;    // [C][out_dim_last_layer]
    double* grad_d_w;   // [total_w]
    double* grad_d_a;   // [total_a]
    float negative_slope = 0.01f; // LeakyReLU slope

     // Start timer
    auto start = std::chrono::high_resolution_clock::now();

     // 4. Initialize parameters and allocate memory
    cudaError_t err;
    // 1. Malloc and memcpy for graph data
    err = cudaMalloc(&d_features, num_nodes * in_dim[0] * sizeof(float));
    //printf("cudaMalloc d_features: %s\n", cudaGetErrorString(err));
    printf("\nsize of input to layer-0 : %u\n", num_nodes * in_dim[0]);

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
        //print output_size for each layer
        printf("\nLayer %d output size: %zu\n", l, output_size);

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

    //========================CREATE HOST MEMORY FOR DEBUG PURPOSE===========================
    //temporary create memories for h_s[l], h_attn_score[l]
    float* h_s[L];
    float* h_attn_score[L];
    float* h_attn_coeff[L];
    float* h_layer_outputs[L];
    float* h_h[L];
    for (int l = 0; l < L; ++l) {
        h_s[l] = (float*)malloc(num_nodes * head[l] * max_degree * out_dim[l] * sizeof(float));
        h_attn_score[l] = (float*)malloc(num_nodes * head[l] * max_degree * sizeof(float));
        h_attn_coeff[l] = (float*)malloc(num_nodes * head[l] * max_degree * sizeof(float));

        size_t output_size;
        if (l == L - 1) {
            // Last layer: average heads, only out_dim[l] per node
            output_size = num_nodes * out_dim[l];
        } else {
            // Non-last layer: concatenate heads
            output_size = num_nodes * head[l] * out_dim[l];
        }

        h_layer_outputs[l] = (float*)malloc(output_size * sizeof(float));
        h_h[l] = (float*)malloc(output_size * sizeof(float));
    }
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
    int w_offset[L]= {0}; // Offsets for weights
    int a_offset[L]= {0}; // Offsets for attention vectors
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
    cudaMalloc(&grad_wo, C * out_dim[L-1] * sizeof(double));
    cudaMemset(grad_wo, 0, C * out_dim[L-1] * sizeof(double)); // Initialize to zero
    cudaMalloc(&grad_d_w, total_w * sizeof(double));
    cudaMemset(grad_d_w, 0, total_w * sizeof(double)); // Initialize to zero
    cudaMalloc(&grad_d_a, total_a * sizeof(double));
    cudaMemset(grad_d_a, 0, total_a * sizeof(double)); // Initialize to zero

    int max_heads = *std::max_element(head, head + L);  // Maximum number of heads across all layers
    int max_out_dim = *std::max_element(out_dim, out_dim + L); // Maximum output dimension across all layers

    //===============INITIALISE XAVIER WEIGHTS=============================
    // 3. Initialize weights and attention vectors using Xavier initialization

    int nthreads = L * max_heads * max_out_dim;
    curandState* d_states;
    cudaMalloc(&d_states, nthreads * sizeof(curandState));
    int blockSize = 128;
    dim3 gridSize((nthreads + blockSize - 1) / blockSize);
    setup_states_kernel<<<gridSize, blockSize>>>(d_states, 12345);
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

    //-------------------------------------COPY INITIAL WEIGHTS TO HOST----------------------------------------------------

    //Copy weights from device to host for saving
    // float* h_w = new float[total_w];
    // float* h_a = new float[total_a];
    // float* h_wo = new float[C * out_dim[L-1]];

    // cudaMemcpy(h_w, d_w, total_w * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_a, d_a, total_a * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_wo, d_wo, C * out_dim[L-1] * sizeof(float), cudaMemcpyDeviceToHost);
    // // //now print these parameters in matrix form for each layer
    // // for (int l = 0; l < L; ++l) {
    // //     printf("\nLayer %d:\n", l);
    // //     printf("\nWeight matrix (d_w):\n");
    // //     for (int od = 0; od < out_dim[l]; ++od) {
    // //         for (int id = 0; id < 2 * in_dim[l]; ++id) {
    // //             printf("%f ", h_w[(l * 1 + 0) * out_dim[l] * 2 * in_dim[l] + od * 2 * in_dim[l] + id]);
    // //         }
    // //         printf("\n");
    // //     }
    // //     printf("\nAttention vector (d_a):\n");
    // //     for (int od = 0; od < out_dim[l]; ++od) {
    // //         printf("%f ", h_a[(l * 1 + 0) * out_dim[l] + od]);
    // //     }
    // //     printf("\n");
    // // }
    // // printf("\nTransformation weight matrix (d_wo):\n");
    // // for (int r = 0; r < C; ++r) {
    // //     for (int od = 0; od < out_dim[L - 1]; ++od) {
    // //         printf("%f ", h_wo[r * out_dim[L - 1] + od]);
    // //     }
    // //     printf("\n");
    // // }


    // // *** ADD THIS SECTION HERE - AFTER WEIGHT INITIALIZATION ***

    // Save weights to files
    // save_array_to_file("weights_w.txt", h_w, total_w);
    // save_array_to_file("weights_a.txt", h_a, total_a);
    // save_array_to_file("weights_wo.txt", h_wo, C * out_dim[L-1]);

    // // Clean up host memory
    // delete[] h_w;
    // delete[] h_a;
    // delete[] h_wo;

    // std::cout << "Weights saved successfully!" << std::endl;

    //--------TOY WEIGHTS---------------------------------------------------

    // //load weights from txt files manually (not using xavier kernel weights)
    // float* h_w;
    // float* h_a;
    // float* h_wo;

    // //load weights
    // int weights_w_len;
    // load_float_array("/content/weights_w.txt", &h_w, weights_w_len);
    // if (weights_w_len != total_w) {
    //     std::cerr << "Invalid weights_w length\n";
    //     return 1;
    // }

    // int weights_a_len;
    // load_float_array("/content/weights_a.txt", &h_a, weights_a_len);
    // if (weights_a_len != total_a) {
    //     std::cerr << "Invalid weights_a length\n";
    //     return 1;
    // }

    // int weights_wo_len;
    // load_float_array("/content/weights_wo.txt", &h_wo, weights_wo_len);
    // if (weights_wo_len != C * out_dim[L-1]) {
    //     std::cerr << "Invalid weights_wo length\n";
    //     return 1;
    // }

    // //print h_w in matrix form
    // printf("\nWeights w:\n");
    // for (int l = 0; l < L; ++l) {
    //     printf("Layer %d:\n", l);
    //     for (int od = 0; od < out_dim[l]; ++od) {
    //         for (int id = 0; id < 2 * in_dim[l]; ++id) {
    //             printf("%f ", h_w[w_offset[l] + od * 2 * in_dim[l] + id]);
    //         }
    //         printf("\n");
    //     }
    // }
    // cudaMemcpy(d_w, h_w, total_w * sizeof(float), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Error copying weights_w to device: %s\n", cudaGetErrorString(err));
    // }

    // cudaMemcpy(d_a, h_a, total_a * sizeof(float), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Error copying weights_a to device: %s\n", cudaGetErrorString(err));
    // }
    // cudaMemcpy(d_wo, h_wo, C * out_dim[L-1] * sizeof(float), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Error copying weights_wo to device: %s\n", cudaGetErrorString(err));
    // }

    //------------------------------------------------------------------------------//

    float* d_loss;      // [num_nodes]
    int*   d_correct;   // [num_nodes]
    cudaMalloc(&d_loss,    num_nodes * sizeof(float));
    cudaMalloc(&d_correct, num_nodes * sizeof(int));

    // Allocate output gradient buffers: one per layer, size: N x in_dim[l]
    float** output_gradients = new float*[L];
    float** grad_d_higher_pre = new float*[L]; // Gradient of pre-activation output of higher layer
    for (int l = 0; l < L; ++l) {
        size_t size1 = (size_t)num_nodes * in_dim[l] * sizeof(float);
        size_t size2 = (size_t)num_nodes * out_dim[l]* head[l] * sizeof(float);
        cudaMalloc(&output_gradients[l], size1);
        cudaMalloc(&grad_d_higher_pre[l], size2);
        cudaMemset(output_gradients[l], 0, size1);  // Initialize to zero
        cudaMemset(grad_d_higher_pre[l], 0, size2); // Initialize to zero
    }
    float* input_gradient_last_layer; // Gradient buffer for the last layer input

    // Allocate input gradient buffer for the last layer: size: N x (last layer no. of heads x out_dim_last layer)
    cudaMalloc(&input_gradient_last_layer, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float));
    cudaMemset(input_gradient_last_layer, 0, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float)); // Initialize to zero

    //**************************|***************************************//
    //**************************|**************************************//

    for (int epoch = 1; epoch <=100; ++epoch) {
        printf("\nEpoch %d\n", epoch);
        // b. Forward pass for each layer
        float* d_layer_inputs = d_features;
        for (int l = 0; l < L; ++l) {
            dim3 grid(num_nodes, head[l]);
            int block = max_degree;
            size_t shared_mem = (max_degree + out_dim[l]) * sizeof(float);
            bool is_last_layer = (l == L - 1);
            const float* d_w_l = d_w + w_offset[l];
            const float* d_a_l = d_a + a_offset[l];
            //print d_layer_inputs address
            //printf("\nd_layer_inputs address for layer %d: %p\n", l, d_layer_inputs);

            gatv2_forward_kernel<<<grid, block, shared_mem>>>(
                num_nodes, in_dim[l], out_dim[l], head[l], d_layer_inputs,
                d_row_ptr, d_col_idx, d_w_l, d_a_l, d_h[l], d_layer_outputs[l],
                attn_score[l], attn_coeff[l], d_leakyrelu[l], d_s[l], is_last_layer, max_degree
            );
            cudaDeviceSynchronize();
            //check for error
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching gatv2_forward_kernel: %s\n", cudaGetErrorString(err));
            }

            d_layer_inputs = d_layer_outputs[l];


            //=========================================PRINT FORWARD PASS OUTPUS START HERE============================

            // //print forward pass for layer outputs. in all print code considered H=1 only.
            // printf("\nForward pass for layer %d completed\n", l);
            // //copy thr d_s to host
            // cudaMemcpyAsync(h_s[l], d_s[l], num_nodes * head[l] * max_degree * out_dim[l] * sizeof(float), cudaMemcpyDeviceToHost);
            // //print h_s[l] for each node in matrix form. for each node matrix dimension will be max_degree x out_dim[l]
            // for(int n=0; n<num_nodes; n++) {
            //     for(int h=0; h<head[l]; h++) {
            //         printf("\ns vector for Node %d, Head %d:\n ", n, h);
            //         for(int m=0; m<max_degree; m++) {
            //             for(int o=0; o<out_dim[l]; o++) {
            //                 printf("%f ", h_s[l][(n * head[l]+h) * max_degree * out_dim[l] + m * out_dim[l] + o]);
            //             }
            //             printf("\n");
            //         }
            //         printf("\n");
            //     }
            // }

            // //copy atten_score
            // cudaMemcpyAsync(h_attn_score[l], attn_score[l], num_nodes * head[l] * max_degree * sizeof(float), cudaMemcpyDeviceToHost);
            // //print h_attn_score[l] for each node in matrix form. for each node matrix dimension will be 1 X max_degree
            // for(int n=0; n<num_nodes; n++) {
            //     for(int h=0; h<head[l]; h++) {
            //         printf("\nAttention score for Node %d, Head %d:\n ", n, h);
            //         for(int m=0; m<max_degree; m++) {
            //             printf("%f ", h_attn_score[l][(n * head[l] + h) * max_degree + m]);
            //         }
            //         printf("\n");
            //     }
            // }

            // //copy attention coefficients
            // cudaMemcpyAsync(h_attn_coeff[l], attn_coeff[l], num_nodes * head[l] * max_degree * sizeof(float), cudaMemcpyDeviceToHost);
            // //print h_attn_coeff[l] for each node in matrix form. for each node matrix dimension will be 1 X max_degree
            // for(int n=0; n<num_nodes; n++) {
            //     for(int h=0; h<head[l]; h++) {
            //         printf("\nAttention coefficients for Node %d, Head %d:\n ", n, h);
            //         for(int m=0; m<max_degree; m++) {
            //             printf("%.17f ", h_attn_coeff[l][(n * head[l] + h) * max_degree + m]);
            //         }
            //         printf("\n");
            //     }
            // }

            // //copy h_layer_outputs and h_h from device
            // if (l == L - 1) {
            //     cudaMemcpyAsync(h_layer_outputs[l], d_layer_outputs[l], num_nodes * out_dim[l] * sizeof(float), cudaMemcpyDeviceToHost);
            //     cudaMemcpyAsync(h_h[l], d_h[l], num_nodes * out_dim[l] * sizeof(float), cudaMemcpyDeviceToHost);
            // } else {
            //     cudaMemcpyAsync(h_layer_outputs[l], d_layer_outputs[l], num_nodes * head[l] * out_dim[l] * sizeof(float), cudaMemcpyDeviceToHost);
            //     cudaMemcpyAsync(h_h[l], d_h[l], num_nodes * head[l] * out_dim[l] * sizeof(float), cudaMemcpyDeviceToHost);
            // }
            // //print h_layer_outputs[l] for each node in matrix form. if it is last layer output then for each node output will be 1 X out_dim[l] else it will be form of head[l] X out_dim[l]
            // for(int n=0; n<num_nodes; n++) {
            //     printf("\nLayer %d output post-non-linearlity for Node %d:\n", l, n);
            //     if (l == L - 1) {
            //         for(int o=0; o<out_dim[l]; o++) {
            //             printf("%.17f ", h_layer_outputs[l][n * out_dim[l] + o]);
            //         }
            //     } else {
            //         for(int h=0; h<head[l]; h++) {
            //             printf("\nHead %d:\n", h);
            //             for(int o=0; o<out_dim[l]; o++) {
            //                 printf("%.17f ", h_layer_outputs[l][(n * head[l] + h) * out_dim[l] + o]);
            //             }
            //         }
            //     }
            //     printf("\n");
            // }

            // // print h_h[l] for each node i.e. pre-non-linearity output of size head[l] X out_dim[l] if not last layer else 1 X out_dim[l]

            // for(int n=0; n<num_nodes; n++) {
            //     printf("\nLayer %d output pre-non-linearlity for Node %d:\n", l, n);
            //     if (l == L - 1) {
            //         for(int o=0; o<out_dim[l]; o++) {
            //             printf("%f ", h_h[l][n * out_dim[l] + o]);
            //         }
            //     } else {
            //         for(int h=0; h<head[l]; h++) {
            //             printf("\nHead %d:\n", h);
            //             for(int o=0; o<out_dim[l]; o++) {
            //                 printf("%.17f ", h_h[l][(n * head[l] + h) * out_dim[l] + o]);
            //             }
            //         }
            //     }
            //     printf("\n");
            // }

            //=========================================PRINTS END HERE============================
        }

        int threads_per_block = 128;
        int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

        gatv2_output_kernel<<<num_blocks, threads_per_block>>>(d_wo,  d_layer_inputs, d_z, d_y, num_nodes, C, out_dim[L - 1]);
        cudaDeviceSynchronize();
        // Check for errors after gatv2_output_kernel
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error launching gatv2_output_kernel: %s\n", cudaGetErrorString(err));
        }

        //=========================================PRINT BLOCK===============================
        // Allocate host memory for d_y
        // float* h_y = (float*)malloc(num_nodes * C * sizeof(float));

        // // Copy d_y from device to host
        // cudaMemcpy(h_y, d_y, num_nodes * C * sizeof(float), cudaMemcpyDeviceToHost);

        // // Print d_y for each node
        // printf("\nOutput probabilities (d_y):\n");
        // for (int n = 0; n < num_nodes; ++n) {
        //     printf("Node %d: ", n);
        //     for (int c = 0; c < C; ++c) {
        //         printf("%.17f ", h_y[n * C + c]);
        //     }
        //     printf("\n");
        // }

        // // Free host memory
        // free(h_y);
        //=========================================PRINT BLOCK END============================
        // c. Calculate loss and accuracy ----

        compute_loss_accuracy_kernel<<<num_blocks, threads_per_block>>>(d_y, d_labels, d_loss, d_correct, num_nodes, C);
        cudaDeviceSynchronize();

        compute_loss_and_accuracy(num_nodes, d_loss, d_correct);

        float* grad_input = input_gradient_last_layer;
        // d. Backward pass
        // 1: Output layer gradient
        size_t shared_memory = (C) * threads_per_block * sizeof(float);
        compute_output_gradients<<<num_blocks, threads_per_block, shared_memory>>>(
            d_y, d_labels, d_h[L-1], d_layer_outputs[L-1], d_wo, grad_wo,
            grad_input, num_nodes, C, out_dim[L-1], head[L-1], negative_slope
        );
        cudaDeviceSynchronize();
        // Check for errors after compute_output_gradients
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error launching compute_output_gradients: %s\n", cudaGetErrorString(err));
        }

        //==========================PRINTING GRADIENTS===========================
        // Allocate dynamic host memory for grad_wo
        // double* h_grad_wo = (double*)malloc(C * out_dim[L-1] * sizeof(double));

        // // Copy grad_wo from device to host
        // cudaMemcpy(h_grad_wo, grad_wo, C * out_dim[L-1] * sizeof(double), cudaMemcpyDeviceToHost);

        // // Print grad_wo for debugging
        // printf("\nGradients of output weights (grad_wo):\n");
        // for (int c = 0; c < C; ++c) {
        //     printf("Class %d: ", c);
        //     for (int od = 0; od < out_dim[L-1]; ++od) {
        //         printf("%.17lf ", h_grad_wo[c * out_dim[L-1] + od]);
        //     }
        //     printf("\n");
        // }

        // // Free host memory
        // free(h_grad_wo);
        
        // // Allocate dynamic host memory for grad_input
        // float* h_grad_input = (float*)malloc(num_nodes * head[L-1] * out_dim[L-1] * sizeof(float));

        // // Copy grad_input from device to host
        // cudaMemcpy(h_grad_input, grad_input, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float), cudaMemcpyDeviceToHost);

        // // Print grad_input for debugging
        // printf("\nInput Gradient for Last Layer pre-nonlinearity(dL_dh):\n");
        // for (int n = 0; n < num_nodes; ++n) {
        //     printf("Node %d:\n", n);
        //     for (int h = 0; h < head[L-1]; ++h) {
        //     //printf("  Head %d: ", h);
        //     for (int od = 0; od < out_dim[L-1]; ++od) {
        //         printf("%.17f ", h_grad_input[(n * head[L-1] + h) * out_dim[L-1] + od]);
        //     }
        //     printf("\n");
        //     }
        // }
        // // Free host memory
        // free(h_grad_input);
        //============================PRINT END=================================================


        // 2: Loop backward through GAT layers
        for (int l = L-1; l >= 0; --l) {
            //printf("\nBackward pass for layer %d\n", l);
            dim3 grid(num_nodes, head[l]);
            int block = max_degree;
            size_t shared_mem = (max_degree + out_dim[l]) * sizeof(float);
            float* d_w_l = d_w + w_offset[l];
            float* d_a_l = d_a + a_offset[l];
            bool is_last_layer = (l == L - 1);
            float* grad_output = output_gradients[l];
            gatv2_layer_backward<<<grid, block, shared_mem>>>(
            num_nodes, in_dim[l], out_dim[l], head[l], d_row_ptr, d_col_idx,
            (l > 0) ? d_layer_outputs[l - 1] : d_features, grad_input, grad_d_higher_pre[l], d_h[l],
            attn_coeff[l], attn_score[l], d_leakyrelu[l], d_w_l, d_a_l,
            d_s[l], grad_d_w + w_offset[l], grad_d_a + a_offset[l],
            grad_output, negative_slope, max_degree, is_last_layer, l, total_w);


            cudaDeviceSynchronize();
            // Check for errors after gatv2_layer_backward
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error launching gatv2_layer_backward: %s\n", cudaGetErrorString(err));
                return 1;
            }

            // Prepare input gradient for next iteration down the stack
            grad_input = grad_output; // input gradient of this layer becomes output gradient of next lower layer

        }


        // // -----> *** 5. PARAMETER UPDATE SECTION —  ***

        //     //---------_CLIP GRADIENTS___________________

        //     clip_grad_norm(grad_d_w, total_w, 5.0f);   //  threshold=5
        //     clip_grad_norm(grad_d_a, total_a, 5.0f);
        //     clip_grad_norm(grad_wo, C * out_dim[L - 1], 5.0f);
        // //_________________________________________________
        float lr = 0.0005f; // Set your learning rate

        int block_size = 256;

        // --- Update d_w ---
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

        cudaDeviceSynchronize();

        //     //================PRINT GRADIENTS FOR DEBUGGING=========================================
        //     //copy to host d_w, d_a, d_wo
        //     double* h_grad_w = new double[total_w];
        //     double* h_grad_a = new double[total_a];
        //     double* h_grad_wo = new double[total_wo];
        //     cudaMemcpy(h_grad_w, grad_d_w, total_w * sizeof(double), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(h_grad_a, grad_d_a, total_a * sizeof(double), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(h_grad_wo, grad_wo, total_wo * sizeof(double), cudaMemcpyDeviceToHost);

        //     // Print the copied values of gradients for debugging purpose for each layer separately. From each layer, print only the first 20 values.
        //     if(epoch == 1) {
        //         for (int l = 0; l < L; ++l) {
        //             printf("\nLayer %d:\n", l);

        //             // Print gradients of weights (h_grad_w) for this layer
        //             printf("Gradients of weights (h_grad_w):\n");
        //             int start_w = w_offset[l];
        //             int end_w = start_w + head[l] * out_dim[l] * 2 * in_dim[l];
        //             for (int i = start_w; i < start_w + 20 && i < end_w; ++i) {
        //                 printf("%.10lf ", h_grad_w[i]);
        //             }
        //             printf("\n");

        //             // Print gradients of attention vectors (h_grad_a) for this layer
        //             printf("Gradients of attention vectors (h_grad_a):\n");
        //             int start_a = a_offset[l];
        //             int end_a = start_a + head[l] * out_dim[l];
        //             for (int i = start_a; i < start_a + 20 && i < end_a; ++i) {
        //                 printf("%.10lf ", h_grad_a[i]);
        //             }
        //             printf("\n");
        //             }

        //             // Print gradients of output weights (h_grad_wo) for the final layer
        //             printf("\nGradients of output weights (h_grad_wo):\n");
        //             for (int i = 0; i < 20 && i < total_wo; ++i) {
        //                 printf("%.10lf ", h_grad_wo[i]);
        //             }
        //         printf("\n");
        //     }

        //     // Compute per-layer gradient norms
        //     printf("Per-Layer Gradient Norms:\n");
        //     for (int l = 0; l < L; ++l) {
        //         int start_w = w_offset[l];
        //         int layer_w_size = head[l] * out_dim[l] * 2 * in_dim[l];
        //         int start_a = a_offset[l];
        //         int layer_a_size = head[l] * out_dim[l];

        //         double norm_w_layer = compute_gradient_norm(&h_grad_w[start_w], layer_w_size);
        //         double norm_a_layer = compute_gradient_norm(&h_grad_a[start_a], layer_a_size);

        //         printf("  Layer %d - w-norms: %.10lf, a-norms: %.10lf\n", l, norm_w_layer, norm_a_layer);
        //     }

        //     // Add gradient norm for output weights (grad_wo)
        //     double norm_wo = compute_gradient_norm(h_grad_wo, total_wo);
        //     printf("Output Layer Gradient Norm:\n");
        //     printf("  grad_wo: %.10lf\n", norm_wo);

        //     // Clean up
        //     delete[] h_grad_w;
        //     delete[] h_grad_a;
        //     delete[] h_grad_wo;

        //     //-------------------------------------------------------------------

        cudaMemset(grad_d_w, 0, total_w * sizeof(double));
        cudaMemset(grad_d_a, 0, total_a * sizeof(double));
        cudaMemset(grad_wo, 0, total_wo * sizeof(double));
        cudaMemset(input_gradient_last_layer, 0, num_nodes * head[L-1] * out_dim[L-1] * sizeof(float)); // Initialize to zero
        //memset the output gradients to zero
        for (int l = 0; l < L; ++l) {
            cudaMemset(output_gradients[l], 0, num_nodes * in_dim[l] * sizeof(float));
            cudaMemset(grad_d_higher_pre[l], 0, num_nodes * out_dim[l] * head[l] * sizeof(float));
        }

    }


    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    std::cout <<  " total time: " << elapsed.count() << " ms" << std::endl;


}

