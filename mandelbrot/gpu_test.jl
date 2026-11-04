using CUDA, KernelAbstractions, Test

@kernel function double_elements!(output, input)
    i = @index(Global, Linear)
    output[i] = 2f0 * input[i]
end

input_cpu = Float32[1, 2, 3, 4, 5]
input_gpu = CuArray(input_cpu)
output_gpu = similar(input_gpu)

backend = KernelAbstractions.get_backend(output_gpu)
double_elements!(backend, 64)(
    output_gpu, input_gpu;
    ndrange=length(input_gpu)
)
KernelAbstractions.synchronize(backend)

result = Array(output_gpu)
@test result == 2f0 .* input_cpu
result
