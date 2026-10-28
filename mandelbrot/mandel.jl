using Test

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

counts = mandelbrot_cpu(32, 100, -2f0, -1.5f0, 3f0)
println(size(counts))
println(extrema(counts))

@test size(counts) == (32, 32)
@test all(c -> 0 <= c <= 100, counts)

# For this window, row 17 / column 17 represents c = -0.5 + 0im.
@test counts[17, 17] == 100

println("CPU image tests passed.")

function mandelbrot_kernel!(counts, max_iters, xmin, ymin, zoom)

end

@test escape_count(0f0, 0f0, 100) == 100
@test escape_count(-1f0, 0f0, 100) == 100
@test escape_count(3f0, 0f0, 100) == 1

println("First-round single-point tests passed.")

@test escape_count(0f0, 0f0, 0) == 0
@test escape_count(2f0, 0f0, 100) == 2

println("Second-round single-point tests passed.")