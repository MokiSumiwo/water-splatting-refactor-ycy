#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

__device__ double log_phi(double value) {
    if (value > -10.) return log(0.5 * erfc(-value * 0.7071067811865475));
    const double inverse_square = 1. / (value * value);
    const double correction = 1. - inverse_square * (1. - inverse_square * (3. - 15. * inverse_square));
    return -0.5 * value * value - log(-value) - 0.9189385332046727 + log(correction);
}

__device__ float normal_transmission(float mean, float variance, float beta) {
    if (variance < 1e-12f) return __expf(-beta * fmaxf(0.f, mean));
    const double sigma = sqrt((double)variance);
    const double z = mean / sigma;
    const double shifted = z - beta * sigma;
    // Use double for cancellation in the log moment, especially for large beta*sigma.
    const double log_value = -(double)beta * mean + 0.5 * (double)beta * beta * variance
        + (double)log_phi(shifted) - (double)log_phi(z);
    return expf(fminf(0.f, (float)log_value));
}

__global__ void transport_kernel(
    int width, int height, const int* ids, const int2* bins,
    const float2* xys, const float3* conics, const float3* colors,
    const float* opacity, const float3* beta, const float* depths,
    const float3* conditional, float* output) {
    const int x = blockIdx.x * 16 + threadIdx.x;
    const int y = blockIdx.y * 16 + threadIdx.y;
    const bool inside = x < width && y < height;
    const int pixel = y * width + x;
    const int tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int2 range = bins[tile];
    const int local = threadIdx.y * 16 + threadIdx.x;
    __shared__ int gids[256];
    __shared__ float3 positions[256], quadratic[256], condition[256];
    __shared__ float center_depth[256];
    bool done = !inside;
    float transmission = 1.f;
    float sums[16] = {0.f};
    const float3 attenuation = inside ? beta[pixel] : make_float3(0.f, 0.f, 0.f);
    for (int start = range.x; start < range.y; start += 256) {
        if (__syncthreads_count(done) == 256) break;
        const int index = start + local;
        if (index < range.y) {
            const int g = ids[index];
            gids[local] = g;
            positions[local] = make_float3(xys[g].x, xys[g].y, opacity[g]);
            quadratic[local] = conics[g];
            condition[local] = conditional[g];
            center_depth[local] = depths[g];
        }
        __syncthreads();
        const int count = min(256, range.y - start);
        for (int k = 0; k < count && !done; ++k) {
            const float3 pos = positions[k], conic = quadratic[k];
            const float dx = pos.x - x, dy = pos.y - y;
            const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
            const float alpha = fminf(0.999f, pos.z * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) continue;
            const float next = transmission * (1.f - alpha);
            if (next <= 1e-4f) { done = true; break; }
            const float weight = alpha * transmission;
            const float3 rgb = colors[gids[k]], stats = condition[k];
            const float depth = center_depth[k];
            const float mean = depth - stats.x * dx - stats.y * dy;
            const float c[3] = {rgb.x, rgb.y, rgb.z};
            const float b[3] = {attenuation.x, attenuation.y, attenuation.z};
            for (int ch = 0; ch < 3; ++ch) {
                const float contribution = weight * c[ch];
                sums[ch] += __expf(-b[ch] * depth) * contribution;
                sums[3 + ch] += __expf(-b[ch] * fmaxf(mean, 0.f)) * contribution;
                sums[6 + ch] += normal_transmission(mean, stats.z, b[ch]) * contribution;
                sums[9 + ch] += normal_transmission(depth, stats.z, b[ch]) * contribution;
                sums[12 + ch] += contribution;
            }
            sums[15] += weight;
            transmission = next;
        }
        __syncthreads();
    }
    if (inside) for (int k = 0; k < 16; ++k) output[16 * pixel + k] = sums[k];
}

torch::Tensor optical_path_forward(
    torch::Tensor ids, torch::Tensor bins, torch::Tensor xys, torch::Tensor conics,
    torch::Tensor colors, torch::Tensor opacity, torch::Tensor beta,
    torch::Tensor depths, torch::Tensor conditional, int height, int width) {
    const c10::cuda::CUDAGuard device_guard(xys.device());
    TORCH_CHECK(xys.is_cuda() && xys.scalar_type() == torch::kFloat32, "CUDA float32 required");
    for (const auto& value : {conics, colors, opacity, beta, depths, conditional})
        TORCH_CHECK(value.device() == xys.device() && value.is_contiguous() && value.scalar_type() == torch::kFloat32,
                    "contiguous CUDA float32 inputs on one device required");
    TORCH_CHECK(xys.is_contiguous() && xys.dim() == 2 && xys.size(1) == 2, "invalid xy shape");
    TORCH_CHECK(colors.sizes() == conics.sizes() && colors.sizes() == conditional.sizes(), "invalid Nx3 shapes");
    TORCH_CHECK(colors.size(0) == xys.size(0) && colors.size(1) == 3, "invalid colors");
    TORCH_CHECK(opacity.numel() == xys.size(0) && depths.numel() == xys.size(0), "invalid per-Gaussian arrays");
    TORCH_CHECK(beta.numel() == height * width * 3, "invalid medium dimensions");
    TORCH_CHECK(ids.is_cuda() && bins.is_cuda() && ids.is_contiguous() && bins.is_contiguous()
                && ids.scalar_type() == torch::kInt32 && bins.scalar_type() == torch::kInt32, "invalid bins");
    auto output = torch::empty({height, width, 16}, xys.options());
    dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
    transport_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        width, height, ids.data_ptr<int>(), (const int2*)bins.data_ptr<int>(),
        (const float2*)xys.data_ptr<float>(), (const float3*)conics.data_ptr<float>(),
        (const float3*)colors.data_ptr<float>(), opacity.data_ptr<float>(), (const float3*)beta.data_ptr<float>(),
        depths.data_ptr<float>(), (const float3*)conditional.data_ptr<float>(), output.data_ptr<float>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "optical path kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optical_path_forward);
}
