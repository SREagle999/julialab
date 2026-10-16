
# Activate the Julia project
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# load packages
using LinearAlgebra
using SparseArrays
using Plots, LaTeXStrings



function some_linear_algebra(b)
  n = length(b)
  A = rand(n, n)
  x = A \ b
  @info "Computation done" A b x norm(A*x-b)
  return x
end


"""
    some_sparse_stuff(n)

This is a docstring for the function `some_sparse_stuff`. This function is
written as a simple demonstration how to set up a sparse matrix of size
`n` × `n` in Julia.
"""
function some_sparse_stuff(n)
  A = spdiagm(0 => ones(n), 1 => -ones(n-1))
end


"""
    some_plotting(n)

This is another docstring. The function `some_plotting` demonstrates how to
create a simple plot of `n` data points using the Julia package Plots.jl.
"""
function some_plotting(n)
  x = range(0.0, 1.0, length = n)
  u = sin.(π .* x) # broadcasting, can also be written as u = @. sin(π * x)
  plot(x, u, xguide = L"x", yguide = L"u", label = "sin")
end


"""
    some_conditionals_and_loops(n)

The function `some_conditionals_and_loops` demonstrates how to write
loops and conditional expressions in Julia. It accepts an integer `n`
and performs some operations, printing stuff to the REPL - and returns
`nothing`.
"""
function some_conditionals_and_loops(n::Integer)
  if n < 10
    println("n = $n < 10")
    1 + 1 # nothing is printed automatically - no need to append ;
    println(2)
  end
  println()

  for i in 1:n
    println("i = ", i)
  end
  println()

  while n^2 > 1
    println("Current value of n: ", n)
    n = n ÷ 2
  end

  return nothing
end
