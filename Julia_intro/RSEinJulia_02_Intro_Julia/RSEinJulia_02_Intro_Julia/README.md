# Introduction to Julia

This code was developed with Julia v1.9.3.


## Introduction to the syntax etc.

The introduction is provided as a Jupyter notebook. To run it locally,
you first need to [install Julia](https://julialang.org/downloads/).
Then, you need to install the Julia package
[IJulia.jl](https://github.com/JuliaLang/IJulia.jl)
in your default environment. You can for example launch Julia in a
terminal and execute

```julia
julia> import Pkg; Pkg.add("IJulia")
```

in the Julia REPL.

If everything is installed, you can open Julia in this dorectory
and start a Jupyter notebook. To do so, you can for example run

```bash
julia -e 'using IJulia; notebook(dir = pwd())'
```

in a terminal. From there, you can open the notebook
`introduction_to_julia_notebook.ipynb`.


## Exercises

We provide some exercises to get you started writing Julia code in a
[Pluto.jl](https://github.com/fonsp/Pluto.jl) notebook, based on the
MIT course [Computational Thinking](https://github.com/mitmath/computational-thinking/).
Run

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate(); import Pluto; Pluto.run()'
```

in this directory. Then, open the Pluto notebook `exercises.jl`.


## Workflow and reproducibility

This is another part of the tutorial on Julia. In a nutshell, here is a
standard workflow you can follow to setup a reproducible environment:

0. Install Revise.jl in you default Julia environment
   ```julia
   using Pkg
   Pkg.activate() # activate the default environment/project
   Pkg.add("Revise")
   ```
1. Create a new folder for your project
2. Open the Julia REPL in this folder
3. Create a Julia environment and add packages you want to use
   ```julia
   using Pkg
   Pkg.activate(".") # use the current folder as project
   Pkg.add(["Plots", "LaTeXStrings"])
   ```
4. Create a new Julia file for your project and let Revise track it
   ```julia
   using Revise
   includet("workflow.jl")
   ```
5. Write code in your Julia file, execute it in the REPL, update, and iterate
   until convergence.
6. Write up instructions how to reproduce your results in a `README.md` file
   (including the Julia version you used)
7. Distribute (an archive of) the project folder including your Julia files
   and the TOML files (`Project.toml`, `Manifest.toml`).


## Further ressources

You can find more introductory material online, e.g.,

- [Julia website](https://julialang.org)
- [Julia Discourse forum](https://discourse.julialang.org)
- [Course “Advanced Topics from Scientific Computing” in Berlin](https://www.wias-berlin.de/people/fuhrmann/AdSciComp-WS2223/)
- [Modern Julia Workflows Website](https://modernjuliaworkflows.github.io) (in progress)
- [JuliaNotes.jl](https://m3g.github.io/JuliaNotes.jl/stable/)
