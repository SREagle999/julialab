using CUDA, KernelAbstractions
using Test, BenchmarkTools, Statistics
using Dates

const WG = (32, 8)   # workgroup size is 2D since ditching local mem

@inline function mandel_step(x, y, y2, cx, cy) # muladd vers cheapens
    x_new  = muladd(x, x, cx - y2)
    y_new  = muladd(x + x, y, cy)
    y2_new = y_new * y_new
    total  = muladd(x_new, x_new, y2_new)
    return x_new, y_new, y2_new, total
end

function nonilp_it_count(cx, cy, max_iters)
    x, y, y2, total = 0f0, 0f0, 0f0, 0f0
    iters = 0
    while (iters < max_iters && total <= 4f0)
        x, y, y2, total = mandel_step(x, y, y2, cx, cy)
        iters += 1
    end
    return iters
end

function ilptwo_it_count(cxa, cya, cxb, cyb, max_iters)
    xa, ya, ya2, ta = 0f0, 0f0, 0f0, 0f0
    itersa = 0
    xb, yb, yb2, tb = 0f0, 0f0, 0f0, 0f0
    itersb = 0

    ona = true
    onb = true
    while ona || onb
        ona = itersa < max_iters && ta <= 4f0
        onb = itersb < max_iters && tb <= 4f0
        if ona
            xa, ya, ya2, ta = mandel_step(xa, ya, ya2, cxa, cya)
            itersa += 1
        end
        if onb
            xb, yb, yb2, tb = mandel_step(xb, yb, yb2, cxb, cyb)
            itersb += 1
        end
    end
    return itersa, itersb
end

# four independent pixels advanced in the same loop
function ilpfour_it_count(cxa, cya, cxb, cyb,
                          cxc, cyc, cxd, cyd,
                          max_iters)
    xa, ya, ya2, ta = 0f0, 0f0, 0f0, 0f0
    xb, yb, yb2, tb = 0f0, 0f0, 0f0, 0f0
    xc, yc, yc2, tc = 0f0, 0f0, 0f0, 0f0
    xd, yd, yd2, td = 0f0, 0f0, 0f0, 0f0
    ia = ib = ic = id = 0

    ona = onb = onc = ond = true
    while ona || onb || onc || ond
        ona = ia < max_iters && ta <= 4f0
        onb = ib < max_iters && tb <= 4f0
        onc = ic < max_iters && tc <= 4f0
        ond = id < max_iters && td <= 4f0
        if ona
            xa, ya, ya2, ta = mandel_step(xa, ya, ya2, cxa, cya)
            ia += 1
        end
        if onb
            xb, yb, yb2, tb = mandel_step(xb, yb, yb2, cxb, cyb)
            ib += 1
        end
        if onc
            xc, yc, yc2, tc = mandel_step(xc, yc, yc2, cxc, cyc)
            ic += 1
        end
        if ond
            xd, yd, yd2, td = mandel_step(xd, yd, yd2, cxd, cyd)
            id += 1
        end
    end
    return ia, ib, ic, id
end
#=
# ILP over row pairs, results staged through a local-memory tile
@kernel function twoilp_shared_mandel_ker!(counts, max_iters, xmin, ymin, zoom)
    trow, col = @index(Global, NTuple)
    local_row, local_col = @index(Local, NTuple)
    N = size(counts, 1)
    tile = @localmem Int32 (32, 8, 2)

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
=#
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

# ILP over 4 adjacent columns per thread
@kernel function fourilp_cols_mandel_ker!(counts, max_iters, xmin, ymin, zoom)
    row, tcol = @index(Global, NTuple)
    N = size(counts, 1)
    ncols = size(counts, 2)

    colA = 4 * tcol - 3
    colB = colA + 1
    colC = colA + 2
    colD = colA + 3
    cy  = ymin + (Float32(row - 1)  / Float32(N)) * zoom
    cxa = xmin + (Float32(colA - 1) / Float32(N)) * zoom
    cxb = xmin + (Float32(colB - 1) / Float32(N)) * zoom
    cxc = xmin + (Float32(colC - 1) / Float32(N)) * zoom
    cxd = xmin + (Float32(colD - 1) / Float32(N)) * zoom
    ca, cb, cc, cd = ilpfour_it_count(cxa, cy, cxb, cy, cxc, cy, cxd, cy, max_iters)

    if colA <= ncols
        counts[row, colA] = ca % eltype(counts)
    end
    if colB <= ncols
        counts[row, colB] = cb % eltype(counts)
    end
    if colC <= ncols
        counts[row, colC] = cc % eltype(counts)
    end
    if colD <= ncols
        counts[row, colD] = cd % eltype(counts)
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

# column-paired ILP, 4 pixels per thread
function run_cols4_ilp!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    fourilp_cols_mandel_ker!(backend, WG)(output, max_iters, xmin, ymin, zoom;
        ndrange=(size(output, 1), cld(size(output, 2), 4)))
    KernelAbstractions.synchronize(backend)
    return nothing
end



@kernel function symmetric_mandel_ker!(counts, max_iters, xmin, ymin, zoom)
    trow, col = @index(Global, NTuple)
    N = size(counts, 1)

    row = trow + 1
    mirror = N + 2 - row

    cx = xmin + (Float32(col - 1) / Float32(N)) * zoom
    cy = ymin + (Float32(row  - 1) / Float32(N)) * zoom

    count = nonilp_it_count(cx, cy, max_iters) % eltype(counts)
    counts[row, col] = count
    if mirror != row
        counts[mirror, col] = count
    end
end

# the one leftover row with no valid mirror — fill it the normal way
@kernel function edge_row_mandel_ker!(counts, max_iters, xmin, ymin, zoom)
    col = @index(Global, Linear)
    N = size(counts, 1)
    cx = xmin + (Float32(col - 1) / Float32(N)) * zoom
    counts[1, col] = nonilp_it_count(cx, ymin, max_iters) % eltype(counts)
end

function run_symmetric!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    N = size(output, 1)
    edge_row_mandel_ker!(backend, 32)(output, max_iters, xmin, ymin, zoom;
                                       ndrange = size(output, 2))
    symmetric_mandel_ker!(backend, WG)(output, max_iters, xmin, ymin, zoom;
                                       ndrange = (cld(N - 1, 2), size(output, 2)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function symmetric_fourilp_mandel_ker!(counts, max_iters, xmin, ymin, zoom)
    trow, tcol = @index(Global, NTuple)
    N = size(counts, 1)
    ncols = size(counts, 2)

    row    = trow + 1
    mirror = N + 2 - row

    colA = 4 * tcol - 3
    colB = colA + 1
    colC = colA + 2
    colD = colA + 3

    cy  = ymin + (Float32(row  - 1) / Float32(N)) * zoom
    cxa = xmin + (Float32(colA - 1) / Float32(N)) * zoom
    cxb = xmin + (Float32(colB - 1) / Float32(N)) * zoom
    cxc = xmin + (Float32(colC - 1) / Float32(N)) * zoom
    cxd = xmin + (Float32(colD - 1) / Float32(N)) * zoom

    ca, cb, cc, cd = ilpfour_it_count(cxa, cy, cxb, cy, cxc, cy, cxd, cy, max_iters)

    if colA <= ncols; counts[row, colA] = ca % eltype(counts); end
    if colB <= ncols; counts[row, colB] = cb % eltype(counts); end
    if colC <= ncols; counts[row, colC] = cc % eltype(counts); end
    if colD <= ncols; counts[row, colD] = cd % eltype(counts); end

    if mirror != row   # skip the self-paired center row
        if colA <= ncols; counts[mirror, colA] = ca % eltype(counts); end
        if colB <= ncols; counts[mirror, colB] = cb % eltype(counts); end
        if colC <= ncols; counts[mirror, colC] = cc % eltype(counts); end
        if colD <= ncols; counts[mirror, colD] = cd % eltype(counts); end
    end
end

function run_symmetric_cols4!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    N, ncols = size(output, 1), size(output, 2)
    edge_row_mandel_ker!(backend, 32)(output, max_iters, xmin, ymin, zoom;
                                       ndrange = ncols)
    symmetric_fourilp_mandel_ker!(backend, WG)(output, max_iters, xmin, ymin, zoom;
                                       ndrange = (cld(N - 1, 2), cld(ncols, 4)))
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

# the ChatGPT TESTING SUITE
N = 1024
mit = 2000
xmin, ymin, zoom = -2f0, -1.5f0, 1/10000#change zoom to pushed katutorial code vers
# write new image to verify
#need to have idle gpu with single laucnh for ilp to make min optimize
# 
# compare to normal CUDA matrix mult

# Outputs
original_output   = CUDA.zeros(Int32, N, N)
cols4_ilp_output   = CUDA.zeros(Int32, N, N)
symmetric_output   = CUDA.zeros(Int32, N, N)
symcols4_output    = CUDA.zeros(Int32, N, N)

# Warm up
run_single_kernel!(original_output, mit, xmin, ymin, zoom)
run_cols4_ilp!(cols4_ilp_output,mit, xmin, ymin, zoom)
run_symmetric!(symmetric_output, mit, xmin, ymin, zoom)
run_symmetric_cols4!(symcols4_output, mit, xmin, ymin, zoom)

# Correctness
@test Array(cols4_ilp_output)  == Array(original_output)
@test Array(symcols4_output)[2:513,:] == Array(cols4_ilp_output)[2:513,:]
@test Array(symcols4_output)[2:513,:] == Array(symcols4_output)[end:-1:513,:]

# Benchmarks
original_trial  = @benchmark run_single_kernel!($original_output, $mit, $xmin, $ymin, $zoom)
cols4_trial     = @benchmark run_cols4_ilp!($cols4_ilp_output, $mit, $xmin, $ymin, $zoom)
symmetric_trial = @benchmark run_symmetric!($symmetric_output, $mit, $xmin, $ymin, $zoom)
symcols4_trial  = @benchmark run_symmetric_cols4!($symcols4_output, $mit, $xmin, $ymin, $zoom)

t0 = median(original_trial).time
for (name, trial) in [("original", original_trial), ("cols4_ilp", cols4_trial),
                       ("symmetric", symmetric_trial), ("symmetric+cols4", symcols4_trial)]
    println(rpad(name, 20), round(median(trial).time / 1e3, digits=3), " μs   ",
            round(t0 / median(trial).time, digits=2), "×")
end

results = [
    "original"        => original_trial,
    "cols4_ilp"        => cols4_trial,
    "symmetric"        => symmetric_trial,
    "symmetric_cols4"  => symcols4_trial,
]
outdir = "performance"
mkpath(outdir)
save_results(joinpath(outdir, "bench_results.txt"), results; baseline = "original",
             extra = "N=$N mit=$mit WG=$WG")
save_csv(joinpath(outdir, "bench_results.csv"), results; baseline = "original")