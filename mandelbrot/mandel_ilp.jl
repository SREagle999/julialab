using CUDA, KernelAbstractions
using Test, BenchmarkTools, Statistics
using Dates

const WG = (8,32)

@kernel function fill_axis!(coords, lo, zoom, Nf)
    i = @index(Global, Linear)
    coords[i] = lo + ((i - 1) / Nf) * zoom
end

function axis_coords(backend, n::Integer, lo::Float32, zoom::Float32)
    coords = KernelAbstractions.allocate(backend, Float32, n)
    Nf = Float32(n)
    fill_axis!(backend, 256)(coords, lo, zoom, Nf; ndrange = n)
    return coords
end

@inline function mandel_step(x, y, y2, cx, cy)
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

@kernel function twoilp_cols_mandel_ker!(counts, max_iters, @Const(xs), @Const(ys))
    row, tcol = @index(Global, NTuple)
    ncols = size(counts, 2)

    col1 = 2 * tcol - 1
    col2 = col1 + 1
    cy  = @inbounds ys[row]
    cx1 = @inbounds xs[min(col1, ncols)]
    cx2 = @inbounds xs[min(col2, ncols)]
    c1, c2 = ilptwo_it_count(cx1, cy, cx2, cy, max_iters)

    if col1 <= ncols
        counts[row, col1] = c1 % eltype(counts)
    end
    if col2 <= ncols
        counts[row, col2] = c2 % eltype(counts)
    end
end

@kernel function fourilp_cols_mandel_ker!(counts, max_iters, @Const(xs), @Const(ys))
    row, tcol = @index(Global, NTuple)
    ncols = size(counts, 2)

    colA = 4 * tcol - 3
    colB = colA + 1
    colC = colA + 2
    colD = colA + 3
    cy  = @inbounds ys[row]
    cxa = @inbounds xs[min(colA, ncols)]
    cxb = @inbounds xs[min(colB, ncols)]
    cxc = @inbounds xs[min(colC, ncols)]
    cxd = @inbounds xs[min(colD, ncols)]
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

@kernel function nonilp_kernel!(counts, max_iters, @Const(xs), @Const(ys))
    row, col = @index(Global, NTuple)
    cx = @inbounds xs[col]
    cy = @inbounds ys[row]
    counts[row, col] = nonilp_it_count(cx, cy, max_iters) % eltype(counts)
end

function run_single_kernel!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    xs = axis_coords(backend, size(output, 2), xmin, zoom)
    ys = axis_coords(backend, size(output, 1), ymin, zoom)
    nonilp_kernel!(backend, WG)(output, max_iters, xs, ys; ndrange=size(output))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function run_ilp_kernel!(kernel, output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    N = size(output, 1)
    xs = axis_coords(backend, size(output, 2), xmin, zoom)
    ys = axis_coords(backend, N, ymin, zoom)
    kernel(backend, WG)(output, max_iters, xs, ys;
                        ndrange=(cld(N, 2), size(output, 2)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function run_cols_ilp!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    xs = axis_coords(backend, size(output, 2), xmin, zoom)
    ys = axis_coords(backend, size(output, 1), ymin, zoom)
    twoilp_cols_mandel_ker!(backend, WG)(output, max_iters, xs, ys;
        ndrange=(size(output, 1), cld(size(output, 2), 2)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function run_cols4_ilp!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    xs = axis_coords(backend, size(output, 2), xmin, zoom)
    ys = axis_coords(backend, size(output, 1), ymin, zoom)
    fourilp_cols_mandel_ker!(backend, WG)(output, max_iters, xs, ys;
        ndrange=(size(output, 1), cld(size(output, 2), 4)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function symmetric_mandel_ker!(counts, max_iters, @Const(xs), @Const(ys))
    trow, col = @index(Global, NTuple)
    N = size(counts, 1)

    row = trow + 1
    mirror = N + 2 - row

    cx = @inbounds xs[col]
    cy = @inbounds ys[min(row, N)]

    count = nonilp_it_count(cx, cy, max_iters) % eltype(counts)
    counts[row, col] = count
    if mirror != row
        counts[mirror, col] = count
    end
end

@kernel function edge_row_mandel_ker!(counts, max_iters, @Const(xs), @Const(ys))
    col = @index(Global, Linear)
    cx = @inbounds xs[col]
    counts[1, col] = nonilp_it_count(cx, ys[1], max_iters) % eltype(counts)
end

function run_symmetric!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    N = size(output, 1)
    xs = axis_coords(backend, size(output, 2), xmin, zoom)
    ys = axis_coords(backend, N, ymin, zoom)
    edge_row_mandel_ker!(backend, 32)(output, max_iters, xs, ys;
                                       ndrange = size(output, 2))
    symmetric_mandel_ker!(backend, WG)(output, max_iters, xs, ys;
                                       ndrange = (cld(N - 1, 2), size(output, 2)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function symmetric_fourilp_mandel_ker!(counts, max_iters, @Const(xs), @Const(ys))
    trow, tcol = @index(Global, NTuple)
    N = size(counts, 1)
    ncols = size(counts, 2)

    row    = trow + 1
    mirror = N + 2 - row

    colA = 4 * tcol - 3
    colB = colA + 1
    colC = colA + 2
    colD = colA + 3

    cy  = @inbounds ys[min(row, N)]
    cxa = @inbounds xs[min(colA, ncols)]
    cxb = @inbounds xs[min(colB, ncols)]
    cxc = @inbounds xs[min(colC, ncols)]
    cxd = @inbounds xs[min(colD, ncols)]

    ca, cb, cc, cd = ilpfour_it_count(cxa, cy, cxb, cy, cxc, cy, cxd, cy, max_iters)

    if colA <= ncols; counts[row, colA] = ca % eltype(counts); end
    if colB <= ncols; counts[row, colB] = cb % eltype(counts); end
    if colC <= ncols; counts[row, colC] = cc % eltype(counts); end
    if colD <= ncols; counts[row, colD] = cd % eltype(counts); end

    if mirror != row
        if colA <= ncols; counts[mirror, colA] = ca % eltype(counts); end
        if colB <= ncols; counts[mirror, colB] = cb % eltype(counts); end
        if colC <= ncols; counts[mirror, colC] = cc % eltype(counts); end
        if colD <= ncols; counts[mirror, colD] = cd % eltype(counts); end
    end
end

function run_symmetric_cols4!(output, max_iters, xmin, ymin, zoom)
    backend = KernelAbstractions.get_backend(output)
    N, ncols = size(output, 1), size(output, 2)
    xs = axis_coords(backend, ncols, xmin, zoom)
    ys = axis_coords(backend, N, ymin, zoom)
    edge_row_mandel_ker!(backend, 32)(output, max_iters, xs, ys;
                                       ndrange = ncols)
    symmetric_fourilp_mandel_ker!(backend, WG)(output, max_iters, xs, ys;
                                       ndrange = (cld(N - 1, 2), cld(ncols, 4)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function report(io::IO, name, trial)
    println(io, "== ", name)
    show(io, MIME("text/plain"), trial)
    println(io, "\n")
end

function save_results(path::AbstractString, results; baseline = first(first(results)),
                      mode = "w", extra = "")
    t0 = minimum(Dict(results)[baseline]).time
    open(path, mode) do io
        println(io, "# ", Dates.now(), "   ", extra)
        for (name, trial) in results
            report(io, name, trial)
        end
        println(io, "== Speedup vs ", baseline, " (by minimum time)")
        for (name, trial) in results
            println(io, rpad(name, 28), round(t0 / minimum(trial).time; digits = 3), "×")
        end
        println(io)
    end
    return path
end

function save_csv(path::AbstractString, results; baseline = first(first(results)))
    t0 = minimum(Dict(results)[baseline]).time
    open(path, "w") do io
        println(io, "name,min_us,median_us,mean_us,max_us,samples,speedup")
        for (name, t) in results
            println(io, join((name, minimum(t).time / 1e3, median(t).time / 1e3,
                              mean(t).time / 1e3, maximum(t).time / 1e3,
                              length(t.times), t0 / minimum(t).time), ","))
        end
    end
    return path
end

N = 1024
mit = 2000

zoom = 1f0 / 10000f0
xmin = -0.743643887f0 - 0.5f0 * zoom
ymin =  0.131825904f0 - 0.5f0 * zoom

original_output  = CUDA.zeros(Int32, N, N)
cols4_ilp_output = CUDA.zeros(Int32, N, N)

run_single_kernel!(original_output, mit, xmin, ymin, zoom)
run_cols4_ilp!(cols4_ilp_output, mit, xmin, ymin, zoom)

@test Array(cols4_ilp_output) == Array(original_output)

original_trial = @benchmark run_single_kernel!($original_output, $mit, $xmin, $ymin, $zoom)
cols4_trial    = @benchmark run_cols4_ilp!($cols4_ilp_output, $mit, $xmin, $ymin, $zoom)

t0 = minimum(original_trial).time
for (name, trial) in [("original", original_trial), ("cols4_ilp", cols4_trial)]
    println(rpad(name, 20), round(minimum(trial).time / 1e3, digits=3), " μs   ",
            round(t0 / minimum(trial).time, digits=2), "×")
end

results = [
    "original"  => original_trial,
    "cols4_ilp" => cols4_trial,
]
outdir = "performance"
mkpath(outdir)
save_results(joinpath(outdir, "bench_results.txt"), results; baseline = "original",
             extra = "N=$N mit=$mit WG=$WG xmin=$xmin ymin=$ymin zoom=$zoom")
save_csv(joinpath(outdir, "bench_results.csv"), results; baseline = "original")