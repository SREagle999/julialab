"""
ChatGPT-generated first test for GPU, specifically here CUDA, backend
"""
using CUDA, KernelAbstractions, Test

@kernel function double_elements!(output, input)
    i = @index(Global, Linear)
    output[i] = 2f0 * input[i]
end

input_cpu = Float32[1, 2, 3, 4, 5]
input_gpu = CuArray(input_cpu)
output_gpu = similar(input_gpu) #same type

backend = KernelAbstractions.get_backend(output_gpu) #goes to CUDA
double_elements!(backend, 64)(
    output_gpu, input_gpu;
    ndrange=length(input_gpu)
)
KernelAbstractions.synchronize(backend)

result = Array(output_gpu)
@test result == 2f0 .* input_cpu
result

println("GPU result: ", result)
println("Test passed.")

"""CuArray stores the array in GPU memory.
@kernel defines the operation each work item performs.
@index(Global, Linear) gives each work item its own index, starting at 1.
ndrange=5 requests five work items; 64 sets the workgroup size.
synchronize waits for completion; Array(...) copies the result back to CPU memory."""
