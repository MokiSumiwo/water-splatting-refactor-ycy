#include "bindings.h"
#include "config.h"
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>

namespace {

__device__ __forceinline__ float accurate_exp(const float value) {
    // Match torch.exp's FP32 accuracy; the reference control path uses
    // torch.exp even though the renderer itself uses the faster __expf.
    return expf(value);
}

__device__ __forceinline__ float clamp01(const float value) {
    return fminf(1.0f, fmaxf(0.0f, value));
}

// H1 deliberately stops at the renderer-local directional sensitivity.  The
// modal projection, evidence, gates, and reconstruction stay in PyTorch so
// their operation order remains the formal reference path.
__global__ void raoc_sensitivity_forward_kernel(
    const float* __restrict__ raw_directions,
    const float* __restrict__ medium_rgb_input,
    const float* __restrict__ medium_bs_input,
    const float* __restrict__ medium_attn_input,
    const float* __restrict__ d_rgb_input,
    const float* __restrict__ d_bs_input,
    const float* __restrict__ d_attn_input,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const unsigned height,
    const unsigned width,
    const unsigned block_width,
    const int num_intersects,
    float* __restrict__ sensitivity_out
) {
    const unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pixels = height * width;
    if (pixel >= pixels) {
        return;
    }

    float d_rgb[3];
    float d_bs[3];
    float d_attn[3];
    float medium_rgb[3];
    float medium_bs[3];
    float medium_attn[3];
    const unsigned medium_offset = pixel * 3;
    float min_attn = 0.0f;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        medium_rgb[c] = medium_rgb_input[medium_offset + c];
        medium_bs[c] = medium_bs_input[medium_offset + c];
        medium_attn[c] = medium_attn_input[medium_offset + c];
        d_rgb[c] = d_rgb_input[medium_offset + c];
        d_bs[c] = d_bs_input[medium_offset + c];
        d_attn[c] = d_attn_input[medium_offset + c];
        min_attn = fminf(min_attn, medium_attn[c]);
    }

    float derivative_rgb[3] = {0.0f, 0.0f, 0.0f};
    float derivative_bs[3] = {0.0f, 0.0f, 0.0f};
    float derivative_attn[3] = {0.0f, 0.0f, 0.0f};
    if (num_intersects > 0) {
        const unsigned row = pixel / width;
        const unsigned col = pixel - row * width;
        const unsigned tiles_x = (width + block_width - 1) / block_width;
        const unsigned tile = (row / block_width) * tiles_x + col / block_width;
        const int2 range = tile_bins[tile];
        float trans_before = 1.0f;
        float prev_depth = 0.0f;
        bool stopped = false;
        float trans_final = 1.0f;
        float medium_factor[3] = {0.0f, 0.0f, 0.0f};
        float bs_derivative[3] = {0.0f, 0.0f, 0.0f};
        float attn_derivative[3] = {0.0f, 0.0f, 0.0f};

        for (int sorted_index = range.x; sorted_index < range.y; ++sorted_index) {
            const int32_t gaussian_id = gaussian_ids_sorted[sorted_index];
            const float2 center = xys[gaussian_id];
            const float2 difference = {
                center.x - static_cast<float>(col),
                center.y - static_cast<float>(row),
            };
            const unsigned gaussian_offset = static_cast<unsigned>(gaussian_id) * 3;
            const float sigma = 0.5f * (
                conics[gaussian_offset + 0] * difference.x * difference.x +
                conics[gaussian_offset + 2] * difference.y * difference.y
            ) + conics[gaussian_offset + 1] * difference.x * difference.y;
            const float alpha = fminf(0.999f, opacities[gaussian_id] * accurate_exp(-sigma));
            const float depth = depths[gaussian_id];
            const bool valid = sigma >= 0.0f && alpha * accurate_exp(-min_attn * depth) >= (1.0f / 255.0f);
            if (!valid || stopped) {
                continue;
            }
            const float next_trans = trans_before * (1.0f - alpha);
            if (next_trans <= 1e-4f) {
                stopped = true;
                continue;
            }

            const float visibility = alpha * trans_before;
            const float exp_bs_prev[3] = {
                accurate_exp(-medium_bs[0] * prev_depth),
                accurate_exp(-medium_bs[1] * prev_depth),
                accurate_exp(-medium_bs[2] * prev_depth),
            };
            const float exp_bs_depth[3] = {
                accurate_exp(-medium_bs[0] * depth),
                accurate_exp(-medium_bs[1] * depth),
                accurate_exp(-medium_bs[2] * depth),
            };
            const unsigned color_offset = gaussian_offset;
            const float exp_attn[3] = {
                accurate_exp(-medium_attn[0] * depth),
                accurate_exp(-medium_attn[1] * depth),
                accurate_exp(-medium_attn[2] * depth),
            };
            medium_factor[0] += trans_before * (exp_bs_prev[0] - exp_bs_depth[0]);
            medium_factor[1] += trans_before * (exp_bs_prev[1] - exp_bs_depth[1]);
            medium_factor[2] += trans_before * (exp_bs_prev[2] - exp_bs_depth[2]);
            bs_derivative[0] += trans_before * (-prev_depth * exp_bs_prev[0] + depth * exp_bs_depth[0]);
            bs_derivative[1] += trans_before * (-prev_depth * exp_bs_prev[1] + depth * exp_bs_depth[1]);
            bs_derivative[2] += trans_before * (-prev_depth * exp_bs_prev[2] + depth * exp_bs_depth[2]);
            attn_derivative[0] -= visibility * colors[color_offset + 0] * exp_attn[0] * depth;
            attn_derivative[1] -= visibility * colors[color_offset + 1] * exp_attn[1] * depth;
            attn_derivative[2] -= visibility * colors[color_offset + 2] * exp_attn[2] * depth;
            trans_final *= (1.0f - alpha);
            prev_depth = fmaxf(prev_depth, depth);
            trans_before = next_trans;
        }

        const float tail_exp[3] = {
            accurate_exp(-medium_bs[0] * prev_depth),
            accurate_exp(-medium_bs[1] * prev_depth),
            accurate_exp(-medium_bs[2] * prev_depth),
        };
        medium_factor[0] += trans_final * tail_exp[0];
        medium_factor[1] += trans_final * tail_exp[1];
        medium_factor[2] += trans_final * tail_exp[2];
        bs_derivative[0] -= trans_final * prev_depth * tail_exp[0];
        bs_derivative[1] -= trans_final * prev_depth * tail_exp[1];
        bs_derivative[2] -= trans_final * prev_depth * tail_exp[2];
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            derivative_rgb[c] = medium_factor[c] * d_rgb[c];
            derivative_bs[c] = medium_rgb[c] * bs_derivative[c] * d_bs[c];
            derivative_attn[c] = attn_derivative[c] * d_attn[c];
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            derivative_rgb[c] = d_rgb[c];
        }
    }

    #pragma unroll
    for (int mode = 0; mode < 9; ++mode) {
        const float action_x = derivative_rgb[0] * raw_directions[mode * 9 + 0]
            + derivative_bs[0] * raw_directions[mode * 9 + 3]
            + derivative_attn[0] * raw_directions[mode * 9 + 6];
        const float action_y = derivative_rgb[1] * raw_directions[mode * 9 + 1]
            + derivative_bs[1] * raw_directions[mode * 9 + 4]
            + derivative_attn[1] * raw_directions[mode * 9 + 7];
        const float action_z = derivative_rgb[2] * raw_directions[mode * 9 + 2]
            + derivative_bs[2] * raw_directions[mode * 9 + 5]
            + derivative_attn[2] * raw_directions[mode * 9 + 8];
        const float norm_squared = action_x * action_x + action_y * action_y + action_z * action_z;
        sensitivity_out[pixel * 9 + mode] = static_cast<float>(sqrt(static_cast<double>(norm_squared)));
    }
}

__global__ void raoc_fused_forward_kernel(
    const float* __restrict__ delta_std,
    const float* __restrict__ basis,
    const float* __restrict__ global_gate,
    const float* __restrict__ local_scale,
    const bool* __restrict__ active,
    const float* __restrict__ raw_medium,
    const float* __restrict__ raw_directions,
    const float* __restrict__ medium_rgb_input,
    const float* __restrict__ medium_bs_input,
    const float* __restrict__ medium_attn_input,
    const float* __restrict__ d_rgb_input,
    const float* __restrict__ d_bs_input,
    const float* __restrict__ d_attn_input,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const unsigned height,
    const unsigned width,
    const unsigned block_width,
    const int num_intersects,
    const float density_bias,
    float* __restrict__ delta_raoc_std,
    float* __restrict__ evidence_out,
    float* __restrict__ local_gate_out,
    float* __restrict__ keep_gate_out,
    float* __restrict__ sensitivity_out
) {
    const unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pixels = height * width;
    if (pixel >= pixels) {
        return;
    }

    float delta[9];
    #pragma unroll
    for (int c = 0; c < 9; ++c) {
        delta[c] = delta_std[pixel * 9 + c];
    }

    float d_rgb[3];
    float d_bs[3];
    float d_attn[3];
    float medium_rgb[3];
    float medium_bs[3];
    float medium_attn[3];
    float min_attn = 0.0f;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const unsigned medium_offset = pixel * 3;
        medium_rgb[0] = medium_rgb_input[medium_offset + 0];
        medium_rgb[1] = medium_rgb_input[medium_offset + 1];
        medium_rgb[2] = medium_rgb_input[medium_offset + 2];
        medium_bs[0] = medium_bs_input[medium_offset + 0];
        medium_bs[1] = medium_bs_input[medium_offset + 1];
        medium_bs[2] = medium_bs_input[medium_offset + 2];
        medium_attn[0] = medium_attn_input[medium_offset + 0];
        medium_attn[1] = medium_attn_input[medium_offset + 1];
        medium_attn[2] = medium_attn_input[medium_offset + 2];
        d_rgb[0] = d_rgb_input[medium_offset + 0];
        d_rgb[1] = d_rgb_input[medium_offset + 1];
        d_rgb[2] = d_rgb_input[medium_offset + 2];
        d_bs[0] = d_bs_input[medium_offset + 0];
        d_bs[1] = d_bs_input[medium_offset + 1];
        d_bs[2] = d_bs_input[medium_offset + 2];
        d_attn[0] = d_attn_input[medium_offset + 0];
        d_attn[1] = d_attn_input[medium_offset + 1];
        d_attn[2] = d_attn_input[medium_offset + 2];
        min_attn = fminf(min_attn, medium_attn[c]);
    }

    float derivative_rgb[3] = {0.0f, 0.0f, 0.0f};
    float derivative_bs[3] = {0.0f, 0.0f, 0.0f};
    float derivative_attn[3] = {0.0f, 0.0f, 0.0f};
    float actions[9][3];

    if (num_intersects > 0) {
        const unsigned row = pixel / width;
        const unsigned col = pixel - row * width;
        const unsigned tiles_x = (width + block_width - 1) / block_width;
        const unsigned tile = (row / block_width) * tiles_x + col / block_width;
        const int2 range = tile_bins[tile];

        float trans_before = 1.0f;
        float prev_depth = 0.0f;
        bool stopped = false;
        float trans_final = 1.0f;
        float medium_factor[3] = {0.0f, 0.0f, 0.0f};
        float bs_derivative[3] = {0.0f, 0.0f, 0.0f};
        float attn_derivative[3] = {0.0f, 0.0f, 0.0f};

        for (int sorted_index = range.x; sorted_index < range.y; ++sorted_index) {
            const int32_t gaussian_id = gaussian_ids_sorted[sorted_index];
            const float2 center = xys[gaussian_id];
            const float2 difference = {
                center.x - static_cast<float>(col),
                center.y - static_cast<float>(row),
            };
            const unsigned gaussian_offset = static_cast<unsigned>(gaussian_id) * 3;
            const float sigma = 0.5f * (
                conics[gaussian_offset + 0] * difference.x * difference.x +
                conics[gaussian_offset + 2] * difference.y * difference.y
            ) + conics[gaussian_offset + 1] * difference.x * difference.y;
            const float alpha = fminf(0.999f, opacities[gaussian_id] * accurate_exp(-sigma));
            const float depth = depths[gaussian_id];
            const bool valid = sigma >= 0.0f && alpha * accurate_exp(-min_attn * depth) >= (1.0f / 255.0f);
            if (!valid || stopped) {
                continue;
            }

            const float next_trans = trans_before * (1.0f - alpha);
            if (next_trans <= 1e-4f) {
                stopped = true;
                continue;
            }

            const float visibility = alpha * trans_before;
            const float exp_bs_prev[3] = {
                accurate_exp(-medium_bs[0] * prev_depth),
                accurate_exp(-medium_bs[1] * prev_depth),
                accurate_exp(-medium_bs[2] * prev_depth),
            };
            const float exp_bs_depth[3] = {
                accurate_exp(-medium_bs[0] * depth),
                accurate_exp(-medium_bs[1] * depth),
                accurate_exp(-medium_bs[2] * depth),
            };
            const float color_x = colors[gaussian_offset + 0];
            const float color_y = colors[gaussian_offset + 1];
            const float color_z = colors[gaussian_offset + 2];
            const float exp_attn[3] = {
                accurate_exp(-medium_attn[0] * depth),
                accurate_exp(-medium_attn[1] * depth),
                accurate_exp(-medium_attn[2] * depth),
            };
            medium_factor[0] += trans_before * (exp_bs_prev[0] - exp_bs_depth[0]);
            medium_factor[1] += trans_before * (exp_bs_prev[1] - exp_bs_depth[1]);
            medium_factor[2] += trans_before * (exp_bs_prev[2] - exp_bs_depth[2]);
            bs_derivative[0] += trans_before * (-prev_depth * exp_bs_prev[0] + depth * exp_bs_depth[0]);
            bs_derivative[1] += trans_before * (-prev_depth * exp_bs_prev[1] + depth * exp_bs_depth[1]);
            bs_derivative[2] += trans_before * (-prev_depth * exp_bs_prev[2] + depth * exp_bs_depth[2]);
            attn_derivative[0] -= visibility * color_x * exp_attn[0] * depth;
            attn_derivative[1] -= visibility * color_y * exp_attn[1] * depth;
            attn_derivative[2] -= visibility * color_z * exp_attn[2] * depth;
            trans_final *= (1.0f - alpha);
            prev_depth = fmaxf(prev_depth, depth);
            trans_before = next_trans;
        }

        const float tail_exp[3] = {
            accurate_exp(-medium_bs[0] * prev_depth),
            accurate_exp(-medium_bs[1] * prev_depth),
            accurate_exp(-medium_bs[2] * prev_depth),
        };
        medium_factor[0] += trans_final * tail_exp[0];
        medium_factor[1] += trans_final * tail_exp[1];
        medium_factor[2] += trans_final * tail_exp[2];
        bs_derivative[0] -= trans_final * prev_depth * tail_exp[0];
        bs_derivative[1] -= trans_final * prev_depth * tail_exp[1];
        bs_derivative[2] -= trans_final * prev_depth * tail_exp[2];
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            derivative_rgb[c] = medium_factor[c] * d_rgb[c];
            derivative_bs[c] = medium_rgb[c] * bs_derivative[c] * d_bs[c];
            derivative_attn[c] = attn_derivative[c] * d_attn[c];
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            derivative_rgb[c] = d_rgb[c];
        }
    }

    #pragma unroll
    for (int mode = 0; mode < 9; ++mode) {
        actions[mode][0] = derivative_rgb[0] * raw_directions[mode * 9 + 0] + derivative_bs[0] * raw_directions[mode * 9 + 3] + derivative_attn[0] * raw_directions[mode * 9 + 6];
        actions[mode][1] = derivative_rgb[1] * raw_directions[mode * 9 + 1] + derivative_bs[1] * raw_directions[mode * 9 + 4] + derivative_attn[1] * raw_directions[mode * 9 + 7];
        actions[mode][2] = derivative_rgb[2] * raw_directions[mode * 9 + 2] + derivative_bs[2] * raw_directions[mode * 9 + 5] + derivative_attn[2] * raw_directions[mode * 9 + 8];
        const float norm_squared =
            actions[mode][0] * actions[mode][0] +
            actions[mode][1] * actions[mode][1] +
            actions[mode][2] * actions[mode][2];
        // Keep the norm rounding aligned with torch.linalg.norm under the
        // extension-wide --use_fast_math build flags.
        sensitivity_out[pixel * 9 + mode] = static_cast<float>(sqrt(static_cast<double>(norm_squared)));
    }

    float coefficients[9];
    float keep[9];
    #pragma unroll
    for (int mode = 0; mode < 9; ++mode) {
        float coefficient = 0.0f;
        #pragma unroll
        for (int c = 0; c < 9; ++c) {
            coefficient = fmaf(delta[c], basis[c * 9 + mode], coefficient);
        }
        coefficients[mode] = coefficient;
        const float evidence = fabsf(coefficient) * sensitivity_out[pixel * 9 + mode];
        const float q = fmaxf(local_scale[mode], 1e-12f);
        const float e2 = evidence * evidence;
        float local = e2 / (e2 + q * q);
        if (!active[mode]) {
            local = 0.0f;
        }
        local = clamp01(local);
        const float global = clamp01(global_gate[mode]);
        float keep_value = 1.0f - (1.0f - global) * (1.0f - local);
        keep_value = fmaxf(keep_value, global);
        keep[mode] = clamp01(keep_value);
        evidence_out[pixel * 9 + mode] = evidence;
        local_gate_out[pixel * 9 + mode] = local;
        keep_gate_out[pixel * 9 + mode] = keep[mode];
    }

    #pragma unroll
    for (int c = 0; c < 9; ++c) {
        float reconstructed = 0.0f;
        #pragma unroll
        for (int mode = 0; mode < 9; ++mode) {
            reconstructed = fmaf(basis[c * 9 + mode], coefficients[mode] * keep[mode], reconstructed);
        }
        delta_raoc_std[pixel * 9 + c] = reconstructed;
    }

    bool zero_local = true;
    bool all_keep = true;
    #pragma unroll
    for (int mode = 0; mode < 9; ++mode) {
        zero_local = zero_local && (local_gate_out[pixel * 9 + mode] == 0.0f);
        all_keep = all_keep && (keep[mode] == 1.0f);
    }
    if (all_keep) {
        #pragma unroll
        for (int c = 0; c < 9; ++c) {
            delta_raoc_std[pixel * 9 + c] = delta[c];
        }
    } else if (zero_local) {
        #pragma unroll
        for (int c = 0; c < 9; ++c) {
            float projected = 0.0f;
            #pragma unroll
            for (int d = 0; d < 9; ++d) {
                float projector = 0.0f;
                #pragma unroll
                for (int mode = 0; mode < 9; ++mode) {
                    projector += basis[c * 9 + mode] * global_gate[mode] * basis[d * 9 + mode];
                }
                projected = fmaf(delta[d], projector, projected);
            }
            delta_raoc_std[pixel * 9 + c] = projected;
        }
    }

}

__global__ void raoc_fused_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ basis,
    const float* __restrict__ keep_gate,
    const unsigned pixels,
    float* __restrict__ grad_input
) {
    const unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) {
        return;
    }
    float modal[9];
    bool all_keep = true;
    #pragma unroll
    for (int mode = 0; mode < 9; ++mode) {
        all_keep = all_keep && (keep_gate[pixel * 9 + mode] == 1.0f);
        float value = 0.0f;
        #pragma unroll
        for (int c = 0; c < 9; ++c) {
            value = fmaf(grad_output[pixel * 9 + c], basis[c * 9 + mode], value);
        }
        modal[mode] = value * keep_gate[pixel * 9 + mode];
    }
    if (all_keep) {
        #pragma unroll
        for (int c = 0; c < 9; ++c) {
            grad_input[pixel * 9 + c] = grad_output[pixel * 9 + c];
        }
        return;
    }
    #pragma unroll
    for (int c = 0; c < 9; ++c) {
        float value = 0.0f;
        #pragma unroll
        for (int mode = 0; mode < 9; ++mode) {
            value = fmaf(modal[mode], basis[c * 9 + mode], value);
        }
        grad_input[pixel * 9 + c] = value;
    }
}

}  // namespace

torch::Tensor raoc_sensitivity_forward_tensor(
    const torch::Tensor &raw_medium,
    const torch::Tensor &raw_directions,
    const torch::Tensor &medium_rgb,
    const torch::Tensor &medium_bs,
    const torch::Tensor &medium_attn,
    const torch::Tensor &d_rgb,
    const torch::Tensor &d_bs,
    const torch::Tensor &d_attn,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width,
    const int num_intersects
) {
    CHECK_INPUT(raw_medium);
    CHECK_INPUT(raw_directions);
    CHECK_INPUT(medium_rgb);
    CHECK_INPUT(medium_bs);
    CHECK_INPUT(medium_attn);
    CHECK_INPUT(d_rgb);
    CHECK_INPUT(d_bs);
    CHECK_INPUT(d_attn);
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    TORCH_CHECK(raw_medium.scalar_type() == torch::kFloat32, "raw_medium must be float32");
    TORCH_CHECK(raw_directions.scalar_type() == torch::kFloat32, "raw_directions must be float32");
    TORCH_CHECK(medium_rgb.scalar_type() == torch::kFloat32 && medium_bs.scalar_type() == torch::kFloat32 && medium_attn.scalar_type() == torch::kFloat32, "medium activations must be float32");
    TORCH_CHECK(d_rgb.scalar_type() == torch::kFloat32 && d_bs.scalar_type() == torch::kFloat32 && d_attn.scalar_type() == torch::kFloat32, "activation derivatives must be float32");
    TORCH_CHECK(raw_medium.ndimension() == 2 && raw_medium.size(1) == 9, "raw_medium must have shape [N, 9]");
    TORCH_CHECK(raw_medium.size(0) == static_cast<int64_t>(img_height) * static_cast<int64_t>(img_width), "raw_medium size must match image dimensions");
    TORCH_CHECK(raw_directions.sizes() == torch::IntArrayRef({9, 9}), "raw_directions must have shape [9, 9]");
    TORCH_CHECK(medium_rgb.numel() == raw_medium.size(0) * 3 && medium_bs.numel() == medium_rgb.numel() && medium_attn.numel() == medium_rgb.numel(), "medium activations must have shape [N, 3]");
    TORCH_CHECK(d_rgb.numel() == medium_rgb.numel() && d_bs.numel() == medium_rgb.numel() && d_attn.numel() == medium_rgb.numel(), "activation derivatives must have shape [N, 3]");
    TORCH_CHECK(xys.ndimension() == 2 && xys.size(1) == 2, "xys must have shape [G, 2]");
    TORCH_CHECK(depths.numel() == xys.size(0) && radii.numel() == xys.size(0), "geometry tensors have inconsistent sizes");
    TORCH_CHECK(conics.sizes() == torch::IntArrayRef({xys.size(0), 3}), "conics must have shape [G, 3]");
    TORCH_CHECK(colors.sizes() == torch::IntArrayRef({xys.size(0), 3}), "colors must have shape [G, 3]");
    TORCH_CHECK(opacities.numel() == xys.size(0), "opacities must have one value per gaussian");
    TORCH_CHECK(gaussian_ids_sorted.scalar_type() == torch::kInt32, "gaussian_ids_sorted must be int32");
    TORCH_CHECK(tile_bins.scalar_type() == torch::kInt32 && tile_bins.ndimension() == 2 && tile_bins.size(1) == 2, "tile_bins must have shape [tiles, 2] and dtype int32");

    torch::Tensor sensitivity = torch::empty({raw_medium.size(0), 9}, raw_medium.options().dtype(torch::kFloat32));
    const unsigned pixels = img_height * img_width;
    raoc_sensitivity_forward_kernel<<<(pixels + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
        raw_directions.data_ptr<float>(), medium_rgb.data_ptr<float>(), medium_bs.data_ptr<float>(),
        medium_attn.data_ptr<float>(), d_rgb.data_ptr<float>(), d_bs.data_ptr<float>(), d_attn.data_ptr<float>(),
        reinterpret_cast<const float2*>(xys.data_ptr<float>()), depths.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(), gaussian_ids_sorted.data_ptr<int32_t>(),
        reinterpret_cast<const int2*>(tile_bins.data_ptr<int>()), img_height, img_width, block_width, num_intersects,
        sensitivity.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sensitivity;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
raoc_fused_forward_tensor(
    const torch::Tensor &delta_std,
    const torch::Tensor &basis,
    const torch::Tensor &global_gate,
    const torch::Tensor &local_scale,
    const torch::Tensor &active,
    const torch::Tensor &raw_medium,
    const torch::Tensor &raw_directions,
    const torch::Tensor &medium_rgb,
    const torch::Tensor &medium_bs,
    const torch::Tensor &medium_attn,
    const torch::Tensor &d_rgb,
    const torch::Tensor &d_bs,
    const torch::Tensor &d_attn,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width,
    const int num_intersects,
    const float density_bias
) {
    CHECK_INPUT(delta_std);
    CHECK_INPUT(basis);
    CHECK_INPUT(global_gate);
    CHECK_INPUT(local_scale);
    CHECK_INPUT(active);
    CHECK_INPUT(raw_medium);
    CHECK_INPUT(raw_directions);
    CHECK_INPUT(medium_rgb);
    CHECK_INPUT(medium_bs);
    CHECK_INPUT(medium_attn);
    CHECK_INPUT(d_rgb);
    CHECK_INPUT(d_bs);
    CHECK_INPUT(d_attn);
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    TORCH_CHECK(delta_std.scalar_type() == torch::kFloat32, "delta_std must be float32");
    TORCH_CHECK(basis.scalar_type() == torch::kFloat32, "basis must be float32");
    TORCH_CHECK(global_gate.scalar_type() == torch::kFloat32, "global_gate must be float32");
    TORCH_CHECK(local_scale.scalar_type() == torch::kFloat32, "local_scale must be float32");
    TORCH_CHECK(raw_medium.scalar_type() == torch::kFloat32, "raw_medium must be float32");
    TORCH_CHECK(raw_directions.scalar_type() == torch::kFloat32, "raw_directions must be float32");
    TORCH_CHECK(medium_rgb.scalar_type() == torch::kFloat32 && medium_bs.scalar_type() == torch::kFloat32 && medium_attn.scalar_type() == torch::kFloat32, "medium activations must be float32");
    TORCH_CHECK(d_rgb.scalar_type() == torch::kFloat32 && d_bs.scalar_type() == torch::kFloat32 && d_attn.scalar_type() == torch::kFloat32, "activation derivatives must be float32");
    TORCH_CHECK(delta_std.ndimension() == 2 && delta_std.size(1) == 9, "delta_std must have shape [N, 9]");
    TORCH_CHECK(basis.sizes() == torch::IntArrayRef({9, 9}), "basis must have shape [9, 9]");
    TORCH_CHECK(global_gate.numel() == 9 && local_scale.numel() == 9 && active.numel() == 9, "RAOC state must have 9 modes");
    TORCH_CHECK(raw_medium.sizes() == delta_std.sizes(), "raw_medium and delta_std must have the same shape");
    TORCH_CHECK(raw_directions.sizes() == torch::IntArrayRef({9, 9}), "raw_directions must have shape [9, 9]");
    TORCH_CHECK(medium_rgb.numel() == delta_std.size(0) * 3 && medium_bs.numel() == medium_rgb.numel() && medium_attn.numel() == medium_rgb.numel(), "medium activations must have shape [N, 3]");
    TORCH_CHECK(d_rgb.numel() == medium_rgb.numel() && d_bs.numel() == medium_rgb.numel() && d_attn.numel() == medium_rgb.numel(), "activation derivatives must have shape [N, 3]");
    TORCH_CHECK(xys.ndimension() == 2 && xys.size(1) == 2, "xys must have shape [G, 2]");
    TORCH_CHECK(depths.numel() == xys.size(0) && radii.numel() == xys.size(0), "geometry tensors have inconsistent sizes");
    TORCH_CHECK(conics.sizes() == torch::IntArrayRef({xys.size(0), 3}), "conics must have shape [G, 3]");
    TORCH_CHECK(colors.sizes() == torch::IntArrayRef({xys.size(0), 3}), "colors must have shape [G, 3]");
    TORCH_CHECK(opacities.numel() == xys.size(0), "opacities must have one value per gaussian");

    const auto options = delta_std.options().dtype(torch::kFloat32);
    const auto output_shape = delta_std.sizes();
    torch::Tensor delta_out = torch::empty(output_shape, options);
    torch::Tensor evidence = torch::empty(output_shape, options);
    torch::Tensor local_gate = torch::empty(output_shape, options);
    torch::Tensor keep_gate = torch::empty(output_shape, options);
    torch::Tensor sensitivity = torch::empty(output_shape, options);
    const unsigned pixels = img_height * img_width;
    raoc_fused_forward_kernel<<<(pixels + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
        delta_std.data_ptr<float>(), basis.data_ptr<float>(), global_gate.data_ptr<float>(),
        local_scale.data_ptr<float>(), active.data_ptr<bool>(), raw_medium.data_ptr<float>(),
        raw_directions.data_ptr<float>(), medium_rgb.data_ptr<float>(),
        medium_bs.data_ptr<float>(), medium_attn.data_ptr<float>(),
        d_rgb.data_ptr<float>(), d_bs.data_ptr<float>(), d_attn.data_ptr<float>(),
        reinterpret_cast<const float2*>(xys.data_ptr<float>()), depths.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(),
        gaussian_ids_sorted.data_ptr<int32_t>(), reinterpret_cast<const int2*>(tile_bins.data_ptr<int>()),
        img_height, img_width, block_width, num_intersects, density_bias,
        delta_out.data_ptr<float>(), evidence.data_ptr<float>(), local_gate.data_ptr<float>(),
        keep_gate.data_ptr<float>(), sensitivity.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(delta_out, evidence, local_gate, keep_gate, sensitivity);
}

torch::Tensor raoc_fused_backward_tensor(
    const torch::Tensor &grad_delta_raoc_std,
    const torch::Tensor &basis,
    const torch::Tensor &keep_gate
) {
    CHECK_INPUT(grad_delta_raoc_std);
    CHECK_INPUT(basis);
    CHECK_INPUT(keep_gate);
    TORCH_CHECK(grad_delta_raoc_std.ndimension() == 2 && grad_delta_raoc_std.size(1) == 9, "gradient must have shape [N, 9]");
    TORCH_CHECK(basis.sizes() == torch::IntArrayRef({9, 9}), "basis must have shape [9, 9]");
    TORCH_CHECK(keep_gate.sizes() == grad_delta_raoc_std.sizes(), "keep_gate and gradient must have the same shape");
    torch::Tensor grad_input = torch::empty_like(grad_delta_raoc_std);
    const unsigned pixels = grad_delta_raoc_std.size(0);
    raoc_fused_backward_kernel<<<(pixels + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
        grad_delta_raoc_std.data_ptr<float>(), basis.data_ptr<float>(), keep_gate.data_ptr<float>(), pixels,
        grad_input.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
}
