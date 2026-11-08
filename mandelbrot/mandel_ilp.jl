using CUDA, KernelAbstractions
using Test, BenchmarkTools, Statistics
using Dates

const WG = (32, 8)   # workgroup size; the shared tile below must match (32, 8, 2)

function nonilp_it_count(cx::Float32, cy::Float32, max_iters::Int)
    x2 = 0f0
    y2 = 0f0
    w = 0f0
    iters::Int = 0
    while (iters < max_iters && x2 + y2 <= 4f0)
        x = x2 - y2 + cx
        y = w - (x2 + y2) + cy
        x2 = x * x
        y2 = y * y
        w = (x + y) * (x + y)
        iters += 1
    end
    return iters
end

function ilptwo_it_count(cx1::Float32, cy1::Float32, cx2::Float32, cy2::Float32, max_iters::Int)
    x2_1, y2_1, w1 = 0f0, 0f0, 0f0
    iters1 = 0
    x2_2, y2_2, w2 = 0f0, 0f0, 0f0
    iters2 = 0

    on1 = true
    on2 = true
    while on1 || on2
        on1 = (iters1 < max_iters && x2_1 + y2_1 <= 4f0)
        on2 = (iters2 < max_iters && x2_2 + y2_2 <= 4f0)
        if on1
            x1 = x2_1 - y2_1 + cx1
            y1 = w1 - (x2_1 + y2_1) + cy1
            x2_1 = x1 * x1
            y2_1 = y1 * y1
            w1 = (x1 + y1) * (x1 + y1)
            iters1 += 1
        end
        if on2
            x2 = x2_2 - y2_2 + cx2
            y2 = w2 - (x2_2 + y2_2) + cy2
            x2_2 = x2 * x2
            y2_2 = y2 * y2
            w2 = (x2 + y2) * (x2 + y2)
            iters2 += 1
        end
    end
    return iters1, iters2
end

# ILP over row pairs, direct stores
@kernel function twoilp_naive_mandel_ker!(counts, max_iters, xmin, ymin, zoom)
    trow, col = @index(Global, NTuple)
    N = size(counts, 1)

    row1 = 2 * trow - 1
    row2 = row1 + 1
    cx  = xmin + (Float32(col - 1) / Float32(N)) * zoom
    cy1 = ymin + (Float32(row1 - 1) / Float32(N)) * zoom
    cy2 = ymin + (Float32(row2 - 1) / Float32(N)) * zoom
    count1, count2 = ilptwo_it_count(cx, cy1, cx, cy2, max_iters)

    if row1 <= N
        counts[row1, col] = count1 % eltype(counts)
    end
    if row2 <= N
        counts[row2, col] = count2 % eltype(counts)
    end
end

# ILP over row pairs, results staged through a local-memory tile
@kernel function twoilp_shared_mandel_ker!(counts, max_iters, xmin, ymin, zoom)
    trow, col = @index(Global, NTuple)
    local_row, local_col = @index(Local, NTuple)
    N = size(counts, 1)
    tile = @localmem Int (32, 8, 2)

    row1 = 2 * trow - 1
    row2 = row1 + 1
    cx  = xmin + (Float32(col - 1) / Float32(N)) * zoom
    cy1 = ymin + (Float32(row1 - 1) / Float32(N)) * zoom
    cy2 = ymin + (Float32(row2 - 1) / Float32(N)) * zoom
    count1, count2 = ilptwo_it_count(cx, cy1, cx, cy2, max_iters)

    tile[local_row, local_col, 1] = count1
    tile[local_row, local_col, 2] = count2
    @synchronize

    if row1 <= N
        counts[row1, col] = tile[local_row, local_col, 1] % eltype(counts)
    end
    if row2 <= N
        counts[row2, col] = tile[local_row, local_col, 2] % eltype(counts)
    end
end

# ILP over column pairs: a warp's threads vary along rows (the fast index in
# column-major storage), so stores stay contiguous
@kernel function twoilp_cols_mandel_ker!(counts, max_iters, xmin, ymin, zoom)
    row, tcol = @index(Global, NTuple)
    N = size(counts, 1)

    col1 = 2 * tcol - 1
    col2 = col1 + 1
    cy  = ymin + (Float32(row - 1)  / Float32(N)) * zoom
    cx1 = xmin + (Float32(col1 - 1) / Float32(N)) * zoom
    cx2 = xmin + (Float32(col2 - 1) / Float32(N)) * zoom
    c1, c2 = ilptwo_it_count(cx1, cy, cx2, cy, max_iters)

    if col1 <= size(counts, 2)
        counts[row, col1] = c1 % eltype(counts)
    end
    if col2 <= size(counts, 2)
        counts[row, col2] = c2 % eltype(counts)
    end
end

# LEGACY SINGLE/NON-ILP
@kernel function nonilp_kernel!(counts, max_iters, xmin, ymin, zoom)
    row, col = @index(Global, NTuple)
    N = size(counts, 1)
    cx = xmin + (Float32(col - 1) / Float32(N)) * zoom
    cy = ymin + (Float32(row - 1) / Float32(N)) * zoom
    counts[row, col] = nonilp_it_count(cx, cy, max_iters) % eltype(counts)
end

# RUNNERS
function run_single_kernel!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    nonilp_kernel!(backend, WG)(output, max_iters, xmin, ymin, zoom; ndrange=size(output))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# row-paired ILP: each thread handles 2 rows
function run_ilp_kernel!(kernel, output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    N = size(output, 1)
    kernel(backend, WG)(output, max_iters, xmin, ymin, zoom;
                        ndrange=(cld(N, 2), size(output, 2)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

run_naive_ilp!(o, m, x, y, z)  = run_ilp_kernel!(twoilp_naive_mandel_ker!,  o, m, x, y, z)
run_shared_ilp!(o, m, x, y, z) = run_ilp_kernel!(twoilp_shared_mandel_ker!, o, m, x, y, z)

# column-paired ILP: each thread handles 2 columns
function run_cols_ilp!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    twoilp_cols_mandel_ker!(backend, WG)(output, max_iters, xmin, ymin, zoom;
        ndrange=(size(output, 1), cld(size(output, 2), 2)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# RESULT WRITERS
function report(io::IO, name, trial)
    println(io, "== ", name)
    show(io, MIME("text/plain"), trial)
    println(io, "\n")
end

function save_results(path::AbstractString, results; baseline = first(first(results)),
                      mode = "w", extra = "")
    t0 = median(Dict(results)[baseline]).time
    open(path, mode) do io
        println(io, "# ", Dates.now(), "   ", extra)
        for (name, trial) in results
            report(io, name, trial)
        end
        println(io, "== Speedup vs ", baseline)
        for (name, trial) in results
            println(io, rpad(name, 28), round(t0 / median(trial).time; digits = 3), "×")
        end
        println(io)
    end
    return path
end

function save_csv(path::AbstractString, results; baseline = first(first(results)))
    t0 = median(Dict(results)[baseline]).time
    open(path, "w") do io
        println(io, "name,min_us,median_us,mean_us,max_us,samples,speedup")
        for (name, t) in results
            println(io, join((name, minimum(t).time / 1e3, median(t).time / 1e3,
                              mean(t).time / 1e3, maximum(t).time / 1e3,
                              length(t.times), t0 / median(t).time), ","))
        end
    end
    return path
end

# TESTING SUITE
N = 8192
mit = 1000
xmin, ymin, zoom = -2f0, -1.5f0, 3f0

original_output   = CUDA.zeros(Int32, N, N)
naive_ilp_output  = CUDA.zeros(Int32, N, N)
shared_ilp_output = CUDA.zeros(Int32, N, N)
cols_ilp_output   = CUDA.zeros(Int32, N, N)

# Warm up (compiles all four kernels)
run_single_kernel!(original_output,   mit, xmin, ymin, zoom)
run_naive_ilp!(naive_ilp_output,      mit, xmin, ymin, zoom)
run_shared_ilp!(shared_ilp_output,    mit, xmin, ymin, zoom)
run_cols_ilp!(cols_ilp_output,        mit, xmin, ymin, zoom)

# Correctness
@test Array(naive_ilp_output)  == Array(original_output)
@test Array(shared_ilp_output) == Array(original_output)
@test Array(cols_ilp_output)   == Array(original_output)

# Benchmarks
println("Original: one pixel per thread")
original_trial = @benchmark run_single_kernel!(
    $original_output, $mit, $xmin, $ymin, $zoom)

println("ILP naive: two row-pixels per thread, direct stores")
naive_trial = @benchmark run_naive_ilp!(
    $naive_ilp_output, $mit, $xmin, $ymin, $zoom)

println("ILP shared: two row-pixels per thread, local-memory tile")
shared_trial = @benchmark run_shared_ilp!(
    $shared_ilp_output, $mit, $xmin, $ymin, $zoom)

println("ILP cols: two column-pixels per thread, coalesced stores")
cols_trial = @benchmark run_cols_ilp!(
    $cols_ilp_output, $mit, $xmin, $ymin, $zoom)

display(original_trial)
display(naive_trial)
display(shared_trial)
display(cols_trial)

t0 = median(original_trial).time
println("Naive ILP speedup:  ", round(t0 / median(naive_trial).time;  digits=2), "×")
println("Shared ILP speedup: ", round(t0 / median(shared_trial).time; digits=2), "×")
println("Cols ILP speedup:   ", round(t0 / median(cols_trial).time;   digits=2), "×")

# Save results
results = [
    "original"   => original_trial,
    "ilp_naive"  => naive_trial,
    "ilp_shared" => shared_trial,
    "ilp_cols"   => cols_trial,
]
save_results("bench_results.txt", results; baseline = "original",
             extra = "N=$N mit=$mit WG=$WG")
save_csv("bench_results.csv", results; baseline = "original")