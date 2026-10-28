using CUDA, KernelAbstractions #we'll be building with CUDA, but the whole idea is that this generalizes
using Test, BenchmarkTools
using Plots

#backend = KernelAbstractions.get_backend( #= FILL =#)

function escape_count(cx::Float32, cy::Float32, max_iters::Int)
    x2 = 0f0
    y2 = 0f0
    w = 0f0
    iters::Int = 0

    while (iters < max_iters && x2 + y2 <= 4f0)
        x = x2 - y2 + cx
        y = w - (x2 + y2) + cy      #both follow from math

        x2 = x * x
        y2 = y * y
        w = (x + y) * (x + y)
        
        iters += 1
    end

    return iters
end

function mandelbrot_cpu(N, max_iters, xmin, ymin, zoom)
    counts = zeros(Int, N, N)

    for col in 1:N
        for row in 1:N
            cx = xmin + (Float32(col - 1) / Float32(N)) * zoom        # 1. Calculate cx using col.
            cy = ymin + (Float32(row - 1) / Float32(N)) * zoom        # 2. Calculate cy using row.
            counts[row,col] = escape_count(cx,cy,max_iters)        # 3. Store the escape count in counts[row, col].
        end
    end

    return counts
end

@kernel function mandelbrot_kernel!(counts, max_iters, xmin, ymin, zoom)
    row, col = @index(Global, NTuple)
    N = size(counts, 1)
    cx = xmin + (Float32(col - 1) / Float32(N)) * zoom        # copy-paste
    cy = ymin + (Float32(row - 1) / Float32(N)) * zoom
    counts[row,col] = escape_count(cx,cy,max_iters)
end

function run_gen_kernel!(output, max_iters, xmin, ymin, zoom) # new pipeline func
    backend = KernelAbstractions.get_backend(output)
    mandelbrot_kernel!(backend,(8,8))(output,max_iters, xmin, ymin, zoom; ndrange=size(output))
    KernelAbstractions.synchronize(backend)

    return nothing
end

@test escape_count(0f0, 0f0, 100) == 100
@test escape_count(-1f0, 0f0, 100) == 100
@test escape_count(3f0, 0f0, 100) == 1
println("First-round single-point tests passed.")


@test escape_count(0f0, 0f0, 0) == 0
@test escape_count(2f0, 0f0, 100) == 2
println("Second-round single-point tests passed.")

counts_cpu = mandelbrot_cpu(32, 100, -2f0, -1.5f0, 3f0)
println(size(counts_cpu))
println(extrema(counts_cpu))
@test size(counts_cpu) == (32, 32)
@test all(c -> 0 <= c <= 100, counts_cpu)
# For this window, row 17 / column 17 represents c = -0.5 + 0im.
@test counts_cpu[17, 17] == 100
println("CPU image tests passed.")

counts_gpu = CUDA.zeros(Int, 32, 32)
gpu_backend = KernelAbstractions.get_backend(counts_gpu)
mandelbrot_kernel!(gpu_backend, (8, 8))(counts_gpu, 100, -2f0, -1.5f0, 3f0; ndrange=size(counts_gpu)) #kernel call
KernelAbstractions.synchronize(gpu_backend)
@test Array(counts_gpu) == counts_cpu
println("GPU matches CPU reference.")

#benchmarking
N = 512
mit = 100
xmin, ymin, zoom = -2f0, -1.5f0, 3f0
cpu_output = zeros(Int,N,N) #init with N, thus func doesn't require N since we can just take size(counts) for range
gpu_output = CUDA.zeros(Int,N,N)

stan_runtime = mandelbrot_cpu(N,mit,xmin,ymin,zoom)


run_gen_kernel!(cpu_output,mit,xmin,ymin,zoom)
run_gen_kernel!(gpu_output,mit,xmin,ymin,zoom) #compile/warmup run first

@btime mandelbrot_cpu(N,mit,xmin,ymin,zoom)
@btime run_gen_kernel!(cpu_output,mit,xmin,ymin,zoom)
@btime run_gen_kernel!(gpu_output,mit,xmin,ymin,zoom)
@test Array(gpu_output) == stan_runtime
@test Array(cpu_output) == stan_runtime
println("GPU and CPU kernels match standard single-thread CPU runtime.")

p = heatmap(stan_runtime; aspect_ratio=:equal, color=:inferno,xlabel="Column",ylabel="Row",title="Mandel iter counts")
savefig(p, "mandelbrot.png")