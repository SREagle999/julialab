### A Pluto.jl notebook ###
# v0.19.30

using Markdown
using InteractiveUtils

# ╔═╡ b572d2c8-556d-4520-a8d9-ce0b49ea3b8c
using PlutoUI

# ╔═╡ 5ba2e5a9-4107-4af8-8cb9-6cd4dd2f0a96
using PlutoTeachingTools

# ╔═╡ a116a67f-bc22-4909-a946-0c8ba5f6a622
using LinearAlgebra

# ╔═╡ d2dafccc-d15f-4e4a-89ea-9bc2c96d637e
using BenchmarkTools

# ╔═╡ f6e18d65-ff9d-46ea-9243-201e1c998dde
struct MyDiagonal{T} <: AbstractMatrix{T}
	diag::Vector{T}
end

# ╔═╡ a1b7c283-05dc-40c7-b28e-e7fcc5eebdda
function Base.size(D::MyDiagonal)
	n = length(D.diag)
	return (n, n)
end

# ╔═╡ a149b57f-05d1-4f72-8a9d-6a8d225e629a
function Base.getindex(D::MyDiagonal, i::Int, j::Int)
	@boundscheck checkbounds(D, i, j)
	# We are safe after the check above and could consider
	# adding `@inbounds` below - but we do not want to dive
	# to deep into such advanced topics right now 🙂
    if i == j
        return D.diag[i]
	else
        return zero(eltype(D))
    end
end

# ╔═╡ a2181260-e6cd-11ea-2a69-8d9d31d1ef0e
md"""
# More Julia

Now, we are ready to dive deeper into Julia and learn a bit more
than just writing simple functions like we would in a numerical 
analysis course 🙂
"""

# ╔═╡ ee2e9533-b0f7-4985-9f15-edbbb52b6046
md"""
#### Initializing packages

_When running this notebook for the first time, this could take up several minutes. Hang in there!_
"""

# ╔═╡ a05d2bc8-7024-11eb-08cb-196543bbb8fd
md"## Creating a Julia struct - using interfaces

Here, we create a new Julia struct representing a diagonal matrix.
"

# ╔═╡ 2ed1b75c-c17c-48a6-8350-3a0370991a92
md"Before being able to use it, we need to define some interface functions. For example, have a look at the [abstract array interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array) in the Julia documentation."

# ╔═╡ 85f2f6cf-7dee-4950-8a1f-e30ad4b5f91d
function Base.setindex!(D::MyDiagonal, v, i::Int, j::Int)
	@boundscheck checkbounds(D, i, j)
	# We are safe after the check above and could consider
	# adding `@inbounds` below
    if i == j
        D.diag[i] = v
    else
        throw(ArgumentError("cannot set off-diagonal entry ($i, $j)"))
    end
    return v
end

# ╔═╡ 642eec59-dbbb-46ef-a578-0c1bcb810208
md"Now, we can use our new struct as every matrix type in Julia."

# ╔═╡ b9947122-6c32-4e3a-940b-b930389e7de0
D1 = MyDiagonal([1, 2, 3, 4, 5])

# ╔═╡ d64d35a9-beaa-4a02-8db8-72af88996f6b
D1[1, 2]

# ╔═╡ 317d1729-8090-41d5-aaaf-eea5e26133d0
D1[3, 3]

# ╔═╡ 42e569d8-5904-4d8e-8024-d5cd288f7f8c
D1[3, 6]

# ╔═╡ 99389368-d6de-4c11-9c67-0dd3210cc43f
D1^2

# ╔═╡ b144cd88-56b0-4445-933c-a52d0e63651d
D1 * [0.0, 0.25, 0.5, 0.75, 1.0]

# ╔═╡ 9f73a339-27b2-4904-9752-4aa6f97b075a
let
	b = rand(size(D1, 2))
	x = D1 \ b
	D1 * x - b
end

# ╔═╡ 50503fb0-6677-4484-b879-78ec7356a4bd
eigvals(D1)

# ╔═╡ 7a4db75a-e20a-48c5-9d10-de0106606642
exp(D1)

# ╔═╡ 57e1b5e8-9ee0-4825-9bec-5413f432fd4d
exp.(D1)

# ╔═╡ 9ef5f1dc-5236-4365-889a-9c622d2d4c43
D1 + D1

# ╔═╡ 70f11e33-c1c8-422a-ab46-4505ed15b81d
md"As you see above, the return type of the addition of two of our diagonal matrices is a standard matrix. We can of course specialize this."

# ╔═╡ 4922ba18-d1d2-40ec-aee7-056c3ce803e6
# Base.:+(D1::MyDiagonal, D2::MyDiagonal) = MyDiagonal(D1.diag + D2.diag)

# ╔═╡ 32bed11a-79d0-4daa-8128-1e4681f2c1af
md"These operations all rely on generic fallbacks implemented in Julia. Thus, they can be less efficient than specialized methods using knowledge about diagonal matrices. For example, let's compare the performance of calculating the eigenvalues of our own struct and the `Diagonal` matrix type implemented in the Julia standard library LinearAlgebra."

# ╔═╡ ff1116bd-7496-42ca-93e2-799e0bc08398
D2 = Diagonal(diag(D1))

# ╔═╡ bc8c5b3d-1511-459d-9a9a-691c9a320960
md"Note that the standard `Diagonal` type has a specialized pretty printing. We could of course write something similar for our own struct."

# ╔═╡ b7b9a6e7-f570-489e-80bc-44ca8bf834b4
eigvals(D2)

# ╔═╡ eea202b2-d89f-422f-9006-5f548f55c6ce
@benchmark eigvals(D1)

# ╔═╡ 2f99de8f-cad5-4e76-be9b-aea5fcb22722
@benchmark eigvals(D2)

# ╔═╡ 3e1eaa18-4c9b-4817-8005-6b08783cdc02
md"We can achieve the same performance improvement for our own struct by specializing `eigvals` appropriately."

# ╔═╡ cd66cb76-72a6-4904-afcd-18ac9f21d196
# LinearAlgebra.eigvals(D::MyDiagonal) = copy(D.diag)

# ╔═╡ ea716adf-7782-44b2-a330-0424c820c713
md"""
## Exercise: Create a struct encoding a one-hot vector

A so-called one-hot vector has all entries zero except one entry unity, e.g.,

```julia
[0, 0, 1, 0, 0, 0, 0]
```

See also the [Wikipedia entry](https://en.wikipedia.org/wiki/One-hot).
"""

# ╔═╡ 099d9776-4da5-4a7b-b92b-1b3351551b83
md"👉 Write a Julia struct encoding such a one-hot vector efficiently."

# ╔═╡ bc99c3cf-8e22-4d3e-b0b2-1790869042b6
# this needs to be written!
struct OneHotVector{p,l} <: AbstractMatrix{l}
	p::Int
	l::Int
end

# ╔═╡ 44585883-ddaf-46db-b30b-2da8b3f9f941


# ╔═╡ deac4d31-c01b-4013-a340-b5072148c3ad
md"👉 You may insert additional cells here to specialize the interface functions."

# ╔═╡ 8ba56725-be08-4562-8a51-ed3be8bc9ea8
md"👉 Create a variable `v` as a `OneHotVector` of length 10 with 1 at position 3."

# ╔═╡ 522983b1-3633-40db-9ddf-6263eb4de027
v = OneHotVector(3,10)

# ╔═╡ ad116d8e-9286-40f8-9c7f-b0be6d3cdea7
if !@isdefined(OneHotVector)
	func_not_defined(:OneHotVector)
elseif !@isdefined(v)
	var_not_defined(:v)
elseif ismissing(v)
	still_missing()
elseif !(v isa OneHotVector)
	keep_working(md"`v` should be a `OneHotVector`.")
elseif !(v isa AbstractVector)
	keep_working(md"`v` should be an `AbstractVector`.")
elseif length(v) != 10
	keep_working(md"`v` does not have the correct size.")
elseif sort(unique(v)) != [0, 1]
	keep_working(md"`v` does not have the correct entries'")
elseif sum(v) != 1
	keep_working(md"`v` does not have the correct entries'")
elseif v[3] != 1
	keep_working(md"`v` does not have the correct entries'")
else
	correct(md"Lookg good!")
end

# ╔═╡ 22b476d3-0338-4cfe-b767-2105fdd61122
md"## Learning more

If you want to learn more about Julia, I recommend the following approach.

1. Read (at least parts of) the Julia documentation, in particular
   - the [style guide](https://docs.julialang.org/en/v1/manual/style-guide/)
   - the [performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
2. Read some good (commented, generic, efficient, ...) Julia code
3. Write your own code and learn how to use packages

When writing Julia code, I typically prefer the following approach

1. Make it work
   - Write a first version that does the job and add tests to make sure it continues to work
2. Make it nice
   - Refactor the code so that it becomes easier to use and more generic
3. Make it fast
   - If required, benchmark the code, discover performance-critical parts, and optimize them as desired

What's nice about Julia is that you can stay in the same language for all of these steps - there is no need to switch to a lower-level language such as C, C++, or Fortran.

"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.3.2"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "7477528e0dffb2b6c3e121c1ab2f71b95aa687d7"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "062c5e1a5bf6ada13db96a4ae4749a4c2234f521"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.9"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "61761f58648aa7217445f24f841839b78c712232"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.3"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "fc8504eca188aaae4345649ca6105806bc584b70"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.37"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "260dc274c1bc2cb839e758588c63d9c8b5e639d1"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.5"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

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

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "ba0dc8a8a67cacac4842631f960c046e4e563675"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.8"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "aea4eede5ab3ee188906d0cf3bbfa36eb543dccc"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.8"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "5d9ab1a4faf25a62bb9d07ef0003396ac258ef1c"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.15"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "5005266de4bfe50e53ff44a5cb5c540b6e47a254"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.0"

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
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "c4171a96893f861e4872b9868f5a7b19da8e9bc8"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.6"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

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

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

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
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─a2181260-e6cd-11ea-2a69-8d9d31d1ef0e
# ╟─ee2e9533-b0f7-4985-9f15-edbbb52b6046
# ╠═b572d2c8-556d-4520-a8d9-ce0b49ea3b8c
# ╠═5ba2e5a9-4107-4af8-8cb9-6cd4dd2f0a96
# ╟─a05d2bc8-7024-11eb-08cb-196543bbb8fd
# ╠═f6e18d65-ff9d-46ea-9243-201e1c998dde
# ╟─2ed1b75c-c17c-48a6-8350-3a0370991a92
# ╠═a1b7c283-05dc-40c7-b28e-e7fcc5eebdda
# ╠═a149b57f-05d1-4f72-8a9d-6a8d225e629a
# ╠═85f2f6cf-7dee-4950-8a1f-e30ad4b5f91d
# ╟─642eec59-dbbb-46ef-a578-0c1bcb810208
# ╠═b9947122-6c32-4e3a-940b-b930389e7de0
# ╠═d64d35a9-beaa-4a02-8db8-72af88996f6b
# ╠═317d1729-8090-41d5-aaaf-eea5e26133d0
# ╠═42e569d8-5904-4d8e-8024-d5cd288f7f8c
# ╠═99389368-d6de-4c11-9c67-0dd3210cc43f
# ╠═b144cd88-56b0-4445-933c-a52d0e63651d
# ╠═9f73a339-27b2-4904-9752-4aa6f97b075a
# ╠═a116a67f-bc22-4909-a946-0c8ba5f6a622
# ╠═50503fb0-6677-4484-b879-78ec7356a4bd
# ╠═7a4db75a-e20a-48c5-9d10-de0106606642
# ╠═57e1b5e8-9ee0-4825-9bec-5413f432fd4d
# ╠═9ef5f1dc-5236-4365-889a-9c622d2d4c43
# ╟─70f11e33-c1c8-422a-ab46-4505ed15b81d
# ╠═4922ba18-d1d2-40ec-aee7-056c3ce803e6
# ╟─32bed11a-79d0-4daa-8128-1e4681f2c1af
# ╠═ff1116bd-7496-42ca-93e2-799e0bc08398
# ╟─bc8c5b3d-1511-459d-9a9a-691c9a320960
# ╠═b7b9a6e7-f570-489e-80bc-44ca8bf834b4
# ╠═d2dafccc-d15f-4e4a-89ea-9bc2c96d637e
# ╠═eea202b2-d89f-422f-9006-5f548f55c6ce
# ╠═2f99de8f-cad5-4e76-be9b-aea5fcb22722
# ╟─3e1eaa18-4c9b-4817-8005-6b08783cdc02
# ╠═cd66cb76-72a6-4904-afcd-18ac9f21d196
# ╟─ea716adf-7782-44b2-a330-0424c820c713
# ╟─099d9776-4da5-4a7b-b92b-1b3351551b83
# ╠═bc99c3cf-8e22-4d3e-b0b2-1790869042b6
# ╠═44585883-ddaf-46db-b30b-2da8b3f9f941
# ╟─deac4d31-c01b-4013-a340-b5072148c3ad
# ╟─8ba56725-be08-4562-8a51-ed3be8bc9ea8
# ╠═522983b1-3633-40db-9ddf-6263eb4de027
# ╟─ad116d8e-9286-40f8-9c7f-b0be6d3cdea7
# ╟─22b476d3-0338-4cfe-b767-2105fdd61122
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
