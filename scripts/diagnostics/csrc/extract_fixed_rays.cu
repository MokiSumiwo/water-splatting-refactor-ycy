#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void extract_kernel(int count, int width, int tile_x, int capacity,
    const int64_t* pixels, const int* ids, const int2* bins, const float2* xys,
    const float3* conics, const float* opacity, int* out_ids, float* out_weights,
    int* out_count) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= count) return;
    const int pixel = pixels[ray];
    const int x = pixel % width, y = pixel / width;
    const int2 range = bins[(y / 16) * tile_x + x / 16];
    float transmission = 1.f;
    int used = 0;
    for (int index = range.x; index < range.y; ++index) {
        const int g = ids[index];
        const float2 center = xys[g];
        const float3 conic = conics[g];
        const float dx = center.x - x, dy = center.y - y;
        const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
        const float alpha = fminf(0.999f, opacity[g] * __expf(-sigma));
        if (sigma < 0.f || alpha < 1.f / 255.f) continue;
        const float next = transmission * (1.f - alpha);
        if (next <= 1e-4f) break;
        if (used < capacity) {
            out_ids[ray * capacity + used] = g;
            out_weights[ray * capacity + used] = alpha * transmission;
        }
        ++used;
        transmission = next;
    }
    out_count[ray] = used;
}

std::vector<torch::Tensor> extract_fixed_rays(torch::Tensor pixels, torch::Tensor ids,
    torch::Tensor bins, torch::Tensor xys, torch::Tensor conics, torch::Tensor opacity,
    int height, int width, int capacity) {
    const c10::cuda::CUDAGuard guard(xys.device());
    TORCH_CHECK(pixels.is_cuda() && pixels.scalar_type() == torch::kInt64 && pixels.is_contiguous(), "int64 CUDA pixels required");
    TORCH_CHECK(ids.is_contiguous() && bins.is_contiguous() && xys.is_contiguous() && conics.is_contiguous() && opacity.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(capacity > 0 && width > 0 && height > 0 && pixels.numel() > 0, "invalid dimensions");
    const int count = pixels.numel();
    auto out_ids = torch::zeros({count, capacity}, ids.options());
    auto weights = torch::zeros({count, capacity}, xys.options());
    auto counts = torch::zeros({count}, ids.options());
    extract_kernel<<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        count, width, (width + 15) / 16, capacity, pixels.data_ptr<int64_t>(), ids.data_ptr<int>(),
        (const int2*)bins.data_ptr<int>(), (const float2*)xys.data_ptr<float>(),
        (const float3*)conics.data_ptr<float>(), opacity.data_ptr<float>(), out_ids.data_ptr<int>(), weights.data_ptr<float>(), counts.data_ptr<int>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ray extraction launch failed");
    return {out_ids, weights, counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &extract_fixed_rays); }
