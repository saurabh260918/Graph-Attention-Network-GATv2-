__global__ void compute_output_gradients(
    const float* d_y,        // [N][C], softmax output
    const int* d_labels,     // [N],   true labels
    const float* d_hL,       // [N][out_dim_L], last layer output (already head-averaged)
    const float* d_wo,       // [C][out_dim_L], output linear W
    float* grad_d_wo,        // [C][out_dim_L], output: grad for W_o
    float* grad_d_hL,        // [N][out_dim_L], output: grad for h_i^(L)
    int N, int C, int out_dim_L)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    // Step 1: compute dL/dz = y_hat - y
    float dL_dz[C];
    for (int c = 0; c < C; ++c) {
        dL_dz[c] = d_y[node*C + c] - (c == d_labels[node] ? 1.0f : 0.0f);
    }

    // Step 2: accumulate grad for W_o: dL/dWo += dL/dz * h_i^(L)^T
    for (int c = 0; c < C; ++c) {
        for (int d = 0; d < out_dim_L; ++d) {
            atomicAdd(&grad_d_wo[c*out_dim_L + d], dL_dz[c] * d_hL[node*out_dim_L + d]);
        }
    }

    // Step 3: backprop to h^(L): dL/dh^(L) = W_o^T * dL/dz, store for next layer
    for (int d = 0; d < out_dim_L; ++d) {
        float sum = 0;
        for (int c = 0; c < C; ++c) {
            sum += d_wo[c*out_dim_L + d] * dL_dz[c];
        }
        grad_d_hL[node*out_dim_L + d] = sum;
    }
}

//.........................................................................................................//

__global__ void gatv2_layer_backward(
    int N, int in_dim, int out_dim, int num_heads,
    const int* d_row_ptr, const int* d_col_idx,
    const float* d_x,         // [N][in_dim]
    const float* d_higher,    // [N][num_heads][out_dim]
    const float* attn_coeff,  // [N][num_heads][max_degree]
    const float* attn_score,  // [N][num_heads][max_degree]
    const float* d_leakyrelu, // [N][num_heads][max_degree][out_dim]
    const float* d_w,         // [num_heads][out_dim][2*in_dim]
    const float* d_a,         // [num_heads][out_dim]
    const float* d_s,         // [N][num_heads][max_degree][out_dim]
    float* grad_w,            // [num_heads][out_dim][2*in_dim]
    float* grad_a,            // [num_heads][out_dim]
    float* grad_x_lower,      // [N][in_dim]
    float negative_slope,
    int max_degree
) {
    int i = blockIdx.x;    // Node index
    int h = blockIdx.y;    // Head index
    int tid = threadIdx.x; // Neighbor offset

    int row_start = d_row_ptr[i];
    int row_end = d_row_ptr[i + 1];
    int deg = row_end - row_start;
    //create a shared memory equal to max_degree
    extern __shared__ float shared_memory[]; //in this i will store dL_d_alpha_ij
    if (tid >= deg) return;

    int jj = row_start + tid;
    int j = d_col_idx[jj];

    // --- Step D.2: dL/d alpha_ij ---
    float dL_d_alpha_ij = 0.0f;
    for (int od = 0; od < out_dim; ++od) {
        float dL_d_h = d_higher[(i * num_heads + h) * out_dim + od];
        for (int id = 0; id < in_dim; ++id) {
            float w_ = d_w[(h*out_dim*2*in_dim) + (od*2*in_dim) + (in_dim + id)];
            dL_d_alpha_ij += dL_d_h * w_ * d_x[j*in_dim + id];
        }
    }
    // Store dL/d alpha_ij in shared memory for later use
    shared_memory[tid] = dL_d_alpha_ij;
    __syncthreads(); // Ensure all threads have written their values


    // --- Step D.4: grad_W direct ---
    float alpha = attn_coeff[(i * num_heads + h) * max_degree + (tid)];
    for (int od = 0; od < out_dim; ++od) {
        float dL_d_h = d_higher[(i * num_heads + h) * out_dim + od];
        for (int id = 0; id < in_dim; ++id) {
            float x_j = d_x[j*in_dim + id];
            atomicAdd(&grad_w[(h*out_dim*2*in_dim) + (od*2*in_dim) + (in_dim + id)], alpha * dL_d_h * x_j);    //here massive sequential addition possible
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

   // For node i as a neighbor of node j, accumulate to grad_x_lower[i][*] as per direct formula
    int offset = -1;
    for (int t = d_row_ptr[j]; t < d_row_ptr[j+1]; ++t) {
        if (d_col_idx[t] == i) {
            offset = t - d_row_ptr[j];
            break;
        }
    }

    float alpha_jh_i = attn_coeff[(j * num_heads + h) * max_degree + offset];
    for (int od = 0; od < out_dim; ++od) {
        float dL_d_hj = d_higher[(j * num_heads + h) * out_dim + od];
        // Right-part of W (W_right: [out_dim][in_dim]), maps neighbor features
        for (int id = 0; id < in_dim; ++id) {
            float W_right = d_w[(h * out_dim * 2 * in_dim) + (od * 2 * in_dim) + (in_dim + id)];
            // atomic add: node i, feature id
            atomicAdd(&grad_x_lower[i * in_dim + id],
                    alpha_jh_i * W_right * dL_d_hj);
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
