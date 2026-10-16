### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ d2f93306-d980-40b1-a891-456627f07bc1
using PlutoUI

# ╔═╡ 4c8a8c1f-fc3f-4422-9e4f-fb8342862de4
using PlutoTeachingTools

# ╔═╡ 14511f03-0bd5-4f97-87f3-186cfa71692a
using BenchmarkTools: @benchmark

# ╔═╡ b11c31aa-2ce3-4116-a828-84fbc14b9d01
using LoopVectorization: LoopVectorization, @turbo

# ╔═╡ 86fc489a-2b4f-4d5b-98ec-9c7599ae62f6
using LinearAlgebra

# ╔═╡ dcdd0086-8f44-11ee-35dd-b32d498715d3
md"""
# More low-level hardware details

In the last lecture, we had a look at some low-level details such as
instruction scheduling, microachitecture, SIMD, and their implications
on performance based on a course from
[Jülich Supercomuting Center](https://gitlab.jsc.fz-juelich.de/beckmann1/FromZeroToHero2019).
"""

# ╔═╡ ef2afd78-9524-4845-a8e8-5050833bb5c1
md"""
In this lecture, we will have a look at some lower-level representations of
our Julia code such as LLVM IR or assembler code. A relatively good introduction
to these topics is available online, e.g.,

- [assembly code](https://mcyoung.xyz/2021/11/29/assembly-1/)
- [LLVM IR](https://mcyoung.xyz/2023/08/01/llvm-ir/)

Their website has more interesting blog posts on related issues such as

- [designing a SIMD algorithm fro base64 decoding](https://mcyoung.xyz/2023/11/27/simd-base64/)

If you want to further understand these topics and look at examples, you can
of course use Julia features such as `@code_native` and `@code_llvm` to explore
examples. If you want to do so for C code, a good recommendation is
[godbolt](https://godbolt.org).

Further resources about the altency and throughput of instructions on different
hardware can be found in the 
[instruction tables](https://www.agner.org/optimize/instruction_tables.pdf) 
on the 
[website of Agner Fog](https://www.agner.org/optimize/).
"""

# ╔═╡ b0632019-2657-4c1f-8ece-b63576c130d4
md"""
#### Initializing packages

_When running this notebook for the first time, this could take up several minutes. Hang in there!_
"""

# ╔═╡ 522ae075-d3f9-4341-aea8-93768365a942
md"""
## A simple reduction example

Remember the example from last lecture --- computing the sum of a vector:

$$\sum_{i=1}^n x_i.$$

Some challenges when thinking about the direct implementation:

- How shall we handle numerical round-off errors?
- Do we require binary reproducibility?
- Is reordering allowed?

The compiler does not know the answers --- you have to decide!
"""

# ╔═╡ b6ce5a47-e0a6-4359-905c-0cbe241cdd96
function sum_naive(x)
	s = zero(eltype(x))
	for xi in x
		s += xi
	end
	return s
end

# ╔═╡ 72cba25c-b98a-47e4-8920-2f9f539ecb1c
@benchmark sum_naive(x) setup=(x = randn(Float64, 20_000))

# ╔═╡ e583e98d-a78e-4d58-a472-740b87fde191
@benchmark sum_naive(x) setup=(x = randn(Float32, 20_000))

# ╔═╡ 6d8782b0-1262-42a2-a592-e34d8fbb1f5b
md"""Let's also consider smaller vectors."""

# ╔═╡ bcfb8dbf-bf53-47fe-9a6c-c56eea9a9f7e
md"""
## Simple SIMD version using plain Julia
"""

# ╔═╡ 6c83b1c2-3115-4000-aeef-3ddbd71f49b2
function sum_simd(x)
	s = zero(eltype(x))
	@simd for i in eachindex(x)
		s += x[i]
	end
	return s
end

# ╔═╡ 74c3c1b8-0e1b-4d83-976e-f1cc0ce883f6
@code_llvm debuginfo=:none sum_simd(randn(10))

# ╔═╡ 52143bde-4dac-4aca-807b-b21cbcf63cfa
@benchmark sum_simd(x) setup=(x = randn(Float64, 20_000))

# ╔═╡ 5d8ac90b-ccf8-45b2-bd39-81a4d53dd7bc
@benchmark sum_simd(x) setup=(x = randn(Float32, 20_000))

# ╔═╡ 6a492a4c-f229-4eee-9965-e5f1c44a7785
md"""
## LoopVectorization.jl

[LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl)
provides the macro `@turbo` performing a lot of low-level optimizations
automatically --- typically with good heuristics.
"""

# ╔═╡ 967e6bd3-e89f-4142-858f-ce5a2acd790e
function sum_turbo(x)
	s = zero(eltype(x))
	@turbo for i in eachindex(x)
	# @turbo unroll=3 for i in eachindex(x)
		s += x[i]
	end
	return s
end

# ╔═╡ 3548ecd5-368e-444a-bda4-08cf38f929bf
@code_llvm debuginfo=:none sum_turbo(randn(10))

# ╔═╡ a8f523ed-ec03-4b07-aef0-15fcb0af553c
@benchmark sum_turbo(x) setup=(x = randn(Float64, 20_000))

# ╔═╡ 3233d560-8665-44a4-980b-77671df870d5
@benchmark sum_turbo(x) setup=(x = randn(Float32, 20_000))

# ╔═╡ 9c1114c2-f47c-4b34-b81b-84214cc74afb
md"""
## How to SIMD: avoiding branches
"""

# ╔═╡ 1c91550f-04e1-4e6e-92dc-346b1eddefb3
function sum_branch(x)
	s = zero(eltype(x))
	@simd for i in eachindex(x)
		if x[i] > 0
			s += sin(x[i])
		else
			s += cos(x[i])
		end
	end
	return s
end

# ╔═╡ efc32ff4-d183-4ab0-ac8f-01dfd75a00d6
@code_llvm debuginfo=:none sum_branch(randn(10))

# ╔═╡ 2d0c1a8d-790c-4ec7-875b-ac0717c10db6
@benchmark sum_branch(x) setup=(x = randn(Float64, 1_000))

# ╔═╡ 89a0eb51-3d67-4d44-85cc-5d51ec63214f
md"👉 Improve the performance of `sum_branch_improved` below!"

# ╔═╡ ee302166-c9dc-476c-b8d5-90d79d68504b
function sum_branch_improved(x)
	s = zero(eltype(x))
	@simd for i in eachindex(x)
		if x[i] > 0
			s += sin(x[i])
		else
			s += cos(x[i])
		end
	end
	return s
end

# ╔═╡ e22f814a-85ab-4430-bc7f-1a95d5bcf425
@code_llvm debuginfo=:none sum_branch_improved(randn(10))

# ╔═╡ 62219fe2-afa4-49ab-b7d2-95d02d4dfe78
@benchmark sum_branch_improved(x) setup=(x = randn(Float64, 1_000))

# ╔═╡ ab392052-7bce-402d-a472-6ed8b5155ad8
md"""
## How to SIMD: avoiding memory writes
"""

# ╔═╡ cfc6c497-f55f-446a-b8fb-86c56d7c9b9d
"""
	mat_transpose_vec_mul!(dst::Vector, A::Matrix, src::Vector)

Compute `A' * src` and store the result in `dst`.
"""
function mat_transpose_vec_mul!(dst::Vector, A::Matrix, src::Vector)
	@assert axes(A, 2) == axes(dst, 1)
	@assert axes(A, 1) == axes(src, 1)
	
	for i in axes(A, 2)
		dst[i] = 0
		@simd for j in axes(A, 1)
			dst[i] += A[j, i] * src[j]
		end
	end
	return dst
end

# ╔═╡ daddbb6d-e7bd-473c-9f73-b49021c3646b
let
	n, m = 50, 100
	src = randn(m)
	A = randn(m, n)
	dst1 = randn(n)
	dst2 = randn(n)
	mat_transpose_vec_mul!(dst1, A, src)
	mul!(dst2, A', src) # dst2 = A' * src
	dst1 ≈ dst2
end

# ╔═╡ 36f1c9e3-4388-40b5-a306-4203158a69a4
let
	n, m = 50, 100
	src = randn(m)
	A = randn(m, n)
	dst = randn(n)
	@benchmark mul!($dst, $(A'), $src)
end

# ╔═╡ 0f8e8d76-de7e-41b5-842f-e60203b3afc3
let
	n, m = 50, 100
	src = randn(m)
	A = randn(m, n)
	dst = randn(n)
	@benchmark mat_transpose_vec_mul!($dst, $A, $src)
end

# ╔═╡ 8ef93528-12e5-4c71-abf2-c8070f35baaa
md"👉 Improve the performance of `mat_transpose_vec_mul_improved!` below!"

# ╔═╡ 7f9bd89e-2b4b-46c8-8712-7ac81785c28b
function mat_transpose_vec_mul_improved!(dst::Vector, A::Matrix, src::Vector)
	@assert axes(A, 2) == axes(dst, 1)
	@assert axes(A, 1) == axes(src, 1)
	
	for i in axes(A, 2)
		dst[i] = 0
		@simd for j in axes(A, 1)
			dst[i] += A[j, i] * src[j]
		end
	end
	return dst
end

# ╔═╡ 177a0616-4890-450e-838d-9a7276c398c7
@code_llvm debuginfo=:none mat_transpose_vec_mul_improved!(
	randn(10), randn(10, 10), randn(10))

# ╔═╡ 66fa7bdd-5bd6-40c4-a835-0921ec52a360
let
	n, m = 50, 100
	src = randn(m)
	A = randn(m, n)
	dst1 = randn(n)
	dst2 = randn(n)
	mat_transpose_vec_mul_improved!(dst1, A, src)
	mul!(dst2, A', src)
	dst1 ≈ dst2
end

# ╔═╡ 9f1d610c-8ca0-413d-8f58-003d0415b2e0
let
	n, m = 50, 100
	src = randn(m)
	A = randn(m, n)
	dst = randn(n)
	@benchmark mat_transpose_vec_mul_improved!($dst, $A, $src)
end

# ╔═╡ 3f39d256-d506-42d3-8ec8-5b6272363e67
md"## Appendix"

# ╔═╡ 9524536b-d632-449f-91d4-b7ac19e7241a
space = html"<br><br><br>";

# ╔═╡ 4d01b0cf-9532-4adc-badc-1903be3986ba
space

# ╔═╡ 71615415-6eb4-4b21-b789-c3e33bd1c848
space

# ╔═╡ db29a3a9-3ba9-4f25-85b5-c6835eecbc5c
space

# ╔═╡ 08932906-4bf3-4bc4-aa95-9148e98f5f13
space

# ╔═╡ b57367ba-896c-47b1-a446-541eef7fd4ef
space

# ╔═╡ cceda9d8-52cf-4cda-918c-2d544f3bde97
space

# ╔═╡ dd4de874-6894-4a23-a67d-97b1d4c94ab8
space

# ╔═╡ 89f70166-2377-4e79-a792-945ab3d30b1b
space

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.3.2"
LoopVectorization = "~0.12.166"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.54"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "e7f19eff284ce94d4a900b15d1fe1990fa95f4ef"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "247efbccf92448be332d154d6ca56b9fcdd93c31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.6.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "e49bce680c109bc86e3e75ebcb15040d6ad9e1d3"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.27"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "0f5648fbae0d015e3abe5867bca2b362f67a5894"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.166"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "c165f205e030208760ebd75b5e1f7706761d9218"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.1"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "6990168abf3fe9a6e34ebb0e05aaaddf6572189e"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.10"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "03fec6800a986d191f64f5c0996b59ed526eda25"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.1"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "7209df901e6ed7489fe9b7aa3e46fb788e15db85"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.65"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─dcdd0086-8f44-11ee-35dd-b32d498715d3
# ╟─ef2afd78-9524-4845-a8e8-5050833bb5c1
# ╟─b0632019-2657-4c1f-8ece-b63576c130d4
# ╠═d2f93306-d980-40b1-a891-456627f07bc1
# ╠═4c8a8c1f-fc3f-4422-9e4f-fb8342862de4
# ╠═14511f03-0bd5-4f97-87f3-186cfa71692a
# ╠═b11c31aa-2ce3-4116-a828-84fbc14b9d01
# ╠═86fc489a-2b4f-4d5b-98ec-9c7599ae62f6
# ╟─4d01b0cf-9532-4adc-badc-1903be3986ba
# ╟─522ae075-d3f9-4341-aea8-93768365a942
# ╠═b6ce5a47-e0a6-4359-905c-0cbe241cdd96
# ╠═72cba25c-b98a-47e4-8920-2f9f539ecb1c
# ╠═e583e98d-a78e-4d58-a472-740b87fde191
# ╟─6d8782b0-1262-42a2-a592-e34d8fbb1f5b
# ╟─71615415-6eb4-4b21-b789-c3e33bd1c848
# ╟─bcfb8dbf-bf53-47fe-9a6c-c56eea9a9f7e
# ╠═6c83b1c2-3115-4000-aeef-3ddbd71f49b2
# ╠═74c3c1b8-0e1b-4d83-976e-f1cc0ce883f6
# ╠═52143bde-4dac-4aca-807b-b21cbcf63cfa
# ╠═5d8ac90b-ccf8-45b2-bd39-81a4d53dd7bc
# ╟─db29a3a9-3ba9-4f25-85b5-c6835eecbc5c
# ╟─6a492a4c-f229-4eee-9965-e5f1c44a7785
# ╠═967e6bd3-e89f-4142-858f-ce5a2acd790e
# ╠═3548ecd5-368e-444a-bda4-08cf38f929bf
# ╠═a8f523ed-ec03-4b07-aef0-15fcb0af553c
# ╠═3233d560-8665-44a4-980b-77671df870d5
# ╟─08932906-4bf3-4bc4-aa95-9148e98f5f13
# ╟─9c1114c2-f47c-4b34-b81b-84214cc74afb
# ╠═1c91550f-04e1-4e6e-92dc-346b1eddefb3
# ╠═efc32ff4-d183-4ab0-ac8f-01dfd75a00d6
# ╠═2d0c1a8d-790c-4ec7-875b-ac0717c10db6
# ╟─89a0eb51-3d67-4d44-85cc-5d51ec63214f
# ╠═ee302166-c9dc-476c-b8d5-90d79d68504b
# ╠═e22f814a-85ab-4430-bc7f-1a95d5bcf425
# ╠═62219fe2-afa4-49ab-b7d2-95d02d4dfe78
# ╟─b57367ba-896c-47b1-a446-541eef7fd4ef
# ╟─ab392052-7bce-402d-a472-6ed8b5155ad8
# ╠═cfc6c497-f55f-446a-b8fb-86c56d7c9b9d
# ╠═daddbb6d-e7bd-473c-9f73-b49021c3646b
# ╠═36f1c9e3-4388-40b5-a306-4203158a69a4
# ╠═0f8e8d76-de7e-41b5-842f-e60203b3afc3
# ╟─8ef93528-12e5-4c71-abf2-c8070f35baaa
# ╠═7f9bd89e-2b4b-46c8-8712-7ac81785c28b
# ╠═177a0616-4890-450e-838d-9a7276c398c7
# ╠═66fa7bdd-5bd6-40c4-a835-0921ec52a360
# ╠═9f1d610c-8ca0-413d-8f58-003d0415b2e0
# ╟─cceda9d8-52cf-4cda-918c-2d544f3bde97
# ╟─dd4de874-6894-4a23-a67d-97b1d4c94ab8
# ╟─89f70166-2377-4e79-a792-945ab3d30b1b
# ╟─3f39d256-d506-42d3-8ec8-5b6272363e67
# ╠═9524536b-d632-449f-91d4-b7ac19e7241a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
