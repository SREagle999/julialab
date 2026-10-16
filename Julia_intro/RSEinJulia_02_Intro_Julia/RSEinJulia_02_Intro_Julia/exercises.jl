### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b572d2c8-556d-4520-a8d9-ce0b49ea3b8c
using PlutoUI

# ╔═╡ 5ba2e5a9-4107-4af8-8cb9-6cd4dd2f0a96
using PlutoTeachingTools

# ╔═╡ d6ee91ea-e750-11ea-1260-31ebf3ec6a9b
using Compose

# ╔═╡ ee2e9533-b0f7-4985-9f15-edbbb52b6046
md"""
#### Initializing packages

_When running this notebook for the first time, this could take up several minutes. Hang in there!_
"""

# ╔═╡ a2181260-e6cd-11ea-2a69-8d9d31d1ef0e
md"""
# Exercises: Getting up and running

First of all, **_welcome to the course!_** 🙂
"""

# ╔═╡ a05d2bc8-7024-11eb-08cb-196543bbb8fd
md"## Exercise 0 - _Making a basic function_

Computing the square of a number is easy -- you just multiply it with itself.

##### Algorithm:

Given: $x$

Output: $x^2$

1. Multiply `x` by `x`"

# ╔═╡ e02f7ea6-7024-11eb-3672-fd59a6cff79b
function basic_square(x)
	return x^2
end

# ╔═╡ 6acef56c-7025-11eb-2524-819c30a75d39
let
	result = basic_square(5)
	if !(result isa Number)
		keep_working(md"`basic_square` did not return a number. Did you forget to write `return`?")
	elseif abs(result - 5*5) < 0.01
		correct()
	else
		keep_working()
	end
end

# ╔═╡ 348cea34-7025-11eb-3def-41bbc16c7512
md"If you want to explore further, we have included a few optional exercises below."

# ╔═╡ ea716adf-7782-44b2-a330-0424c820c713
md"""
## Exercise 1 - Manipulating vectors
"""

# ╔═╡ 154d5ac6-9815-4c32-bd6b-631042a4c5af
example_vector = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.7, 0.0, 0.7, 0.9]

# ╔═╡ ac07d216-a2be-4761-80a4-e43a95a4c73a
md"""
#### Exercise 1.1
👉 Make a random vector `random_vect` of length 10 using the `rand` function.
"""

# ╔═╡ d5e8eba8-4d4f-4f18-b8ac-dc55fc4643ce
random_vect = rand(10)

# ╔═╡ ad116d8e-9286-40f8-9c7f-b0be6d3cdea7
if !@isdefined(random_vect)
	not_defined(:random_vect)
elseif ismissing(random_vect)
	still_missing()
elseif !(random_vect isa Vector)
	keep_working(md"`random_vect` should be a `Vector`.")
elseif eltype(random_vect) != Float64
	almost(md"""
		You generated a vector of random integers. For the remaining exercises, we want a vector of `Float64` numbers.

		The (optional) first argument to `rand` specifies the **type** of elements to generate. For example: `rand(Bool, 10)` generates 10 values that are either `true` or `false`. (Try it!)
		""")
elseif length(random_vect) != 10
	keep_working(md"`random_vect` does not have the correct size.")
elseif length(Set(random_vect)) != 10
	keep_working(md"`random_vect` is not 'random enough'")
else
	correct(md"Well done! You can run your code again to generate a new vector!")
end

# ╔═╡ c2d9ba9e-e9a5-400e-8a47-92415d08fef2
md"""
You can find out more about any function (like `rand`) by clicking on the Live Docs in the bottom right of this Pluto window, and typing a function name in the top.

![image](https://user-images.githubusercontent.com/6933510/107848812-c934df80-6df6-11eb-8a32-663d802f5d11.png)


![image](https://user-images.githubusercontent.com/6933510/107848846-0f8a3e80-6df7-11eb-818a-7271ecb9e127.png)

We recommend that you leave the window open while you work on Julia code. It will continually look up documentation for anything you type!

#### Help, I don't see the Live Docs!

Try the following:

🙋 **Are you viewing a static preview?** The Live Docs only work if you _run_ the notebook. If you are reading this on our course website, then click the button in the top right to run the notebook.

🙋 **Is your screen too small?** Try resizing your window or zooming out.
""" |> hint

# ╔═╡ 6a19dc43-7cf3-40a9-9275-5434d75942ff
md"#### Exercise 1.2
👉 Make a function `my_sum` using a `for` loop, which computes the total of a vector of numbers."

# ╔═╡ a9b3bef1-f400-4352-b02c-743caf51335f
function my_sum(xs)
	sum = 0
	for i in xs
		sum += i
	end
	return sum
end

# ╔═╡ 32c0ed9d-0248-4812-9873-81cba8efb621
my_sum([1, 2, 3])

# ╔═╡ bcc0bead-bde4-4b72-80ba-9803b70f517b
if !@isdefined(my_sum)
	not_defined(:my_sum)
else
	let
		result = my_sum([1,2,3])
		if ismissing(result)
			still_missing()
		elseif isnothing(result)
			keep_working(md"Did you forget to write `return`?")
		elseif result != 6
			keep_working()
		else
			correct()
		end
	end
end

# ╔═╡ b64a7a85-7855-4448-9906-1f4a64d497b6
md"#### Exercise 1.3
👉 Use your `my_sum` function to write a function `my_mean`, which computes the mean/average of a vector of numbers."

# ╔═╡ 9ba4b355-1e61-47b9-ac45-ade50dd017c3
function my_mean(xs)
	mean = 0
	for i in xs
		mean += i
	end
	return mean/length(xs)
end

# ╔═╡ e4fe747c-26e4-4b11-b14e-fab4e2d536ea
if !@isdefined(my_mean)
	func_not_defined(:my_mean)
else
	let
		result = my_mean([1, 2, 3])
		if ismissing(result)
			still_missing()
		elseif isnothing(result)
			keep_working(md"Did you forget to write `return`?")
		elseif result != 2
			keep_working()
		else
			correct()
		end
	end
end

# ╔═╡ d4de36a4-caff-4a68-a3ff-a49f3cc446ce
md"👉 Define `m` to be the mean of `random_vect`."

# ╔═╡ 1bd97b90-6667-4be8-9e1a-8a4623ca74d2
m = my_mean(random_vect)

# ╔═╡ bdf1bdc5-d3e8-41de-9416-1c31e0d0965a
if !@isdefined(m)
	not_defined(:m)
elseif ismissing(m)
	still_missing()
elseif !(m isa Number)
	keep_working(md"`m` should be a number.")
elseif m != my_mean(random_vect)
	keep_working()
else
	correct()
end

# ╔═╡ a18a6c6c-3c70-4655-bcf6-e0d1a6119fbd
md"""#### Exercise 1.4
👉 Write a function `demean`, which takes a vector `xs` and subtracts the mean from each value in `xs`. Use your `my_mean` function!"""

# ╔═╡ cf83ef0c-665b-4f1e-bbf3-8ef04c6ab2ac
md"""
> ### Note about _mutation_
> There are two ways to think about this exercise, you could _modify_ the original vector, or you can _create a new vector_. We often prefer the second version, so that the original data is preserved. We generally only use code of the first variant in the most performance-sensitive parts of a program, as it requires more care to write and use correctly. _**Be careful not to get carried away in optimizing code**, especially when learning a new language!_
>
> There is a convention among Julians that functions that modify their argument have a `!` in their name. For example, `sort(x)` returns a sorted _copy_ of `x`, while `sort!(x)` _modifies_ `x` to be sorted.
>
> #### Tips for writing non-mutating code
> 1. _Rewriting_ an existing mutating function to be non-mutating can feel like a 'tedious' and 'inefficient' process. Often, instead of trying to **rewrite** a mutating function, it's best to take a step back and try to think of your problem as _constructing something new_. Instead of a `for` loop, it might make more sense to use **descriptive** primitives like [broadcasting with the dot syntax](https://docs.julialang.org/en/v1/manual/functions/#man-vectorized) (also for [math operators](https://docs.julialang.org/en/v1/manual/mathematical-operations/#man-dot-operators)), and [map and filter](https://www.youtube.com/watch?v=_O-HBDZMLrM).
>
>
> 2. If a mutating algorithm makes the most sense for your problem, then you can first use `copy` to create a copy of an array, and then modify that copy.
>
> We will cover this topic more in the later exercises!

"""

# ╔═╡ f7240706-0bb8-4740-bdf4-d796b9eb647d
function demean(xs)
	return xs .- my_mean(xs)
end

# ╔═╡ 828aab16-921a-447b-b14e-6c5a2cfffdff
test_vect = let

	# feel free to change your test case here!
	to_create = [-1.0, -1.5, 8.5]


	####
	# this cell is a bit funky to deal with a common pitfall from last year
	# it regenerates the vector if you accidentally wrote a mutating function

	# don't worry about how it works for this exercise!

	demean
	to_create
end

# ╔═╡ 7b85f2c8-0cab-4382-9ee6-f8bcd3bc9da8
md"To verify our function, let's check that the mean of the `demean(test_vect)` is 0: (_Due to floating-point round-off error it may *not* be *exactly* 0._)"

# ╔═╡ a90ea392-291b-4357-9b2f-7b450a6eb19d
demeaned_test_vect = demean(test_vect)

# ╔═╡ d74266d2-1d71-4a71-af13-b0c07bb73343
my_mean(demeaned_test_vect)

# ╔═╡ 53ce3ab7-3edd-4a57-920b-84c382bc8de7
if !@isdefined(my_mean)
	func_not_defined(:my_mean)
else
	let
		input = Float64[1, 2, 3]
		result = demean(input)

		if input === result
			almost(md"""
			It looks like you **modified** `xs` inside the function.

			It is preferable to avoid mutation inside functions, because you might want to use the original data again. For example, applying `demean` to a dataset of sensor readings would **modify** the original data, and the rest of your analysis would be erroneous.

			""")
		elseif ismissing(result)
			still_missing()
		elseif isnothing(result)
			keep_working(md"Did you forget to write `return`?")
		elseif !(result isa AbstractVector) || length(result) != 3
			keep_working(md"Return a vector of the same size as `xs`.")
		elseif abs(sum(result) / 3) < 1e-10
			correct()
		else
			keep_working()
		end
	end
end

# ╔═╡ b2705e7f-61d8-421b-bda5-286cf0fb879d
md"""
#### Exercise 1.5

👉 Generate a vector of 100 elements. Where:
- the center 20 elements are set to `1`, and
- all other elements are `0`.
"""

# ╔═╡ c98e7e36-cbc7-4235-8c27-43a5fc5fddca
function create_bar()
	vec = zeros(100);
	vec[41:60] = ones(20);
	return vec
end

# ╔═╡ c12b18d2-76dd-4ea1-a289-8f20ff791f1b
create_bar()

# ╔═╡ d6f38ac5-e88e-44cd-be62-2400927fe7bc
if !@isdefined(create_bar)
	not_defined(:create_bar)
else
	let
		result = create_bar()
		if ismissing(result)
			still_missing()
		elseif isnothing(result)
			keep_working(md"Did you forget to write `return`?")
		elseif !(result isa Vector) || length(result) != 100
			keep_working(md"The result should be a `Vector` with 100 elements.")
		elseif result[[1,50,100]] != [0,1,0]
			keep_working()
		else
			correct()
		end
	end
end

# ╔═╡ 339c2d5c-e6ce-11ea-32f9-714b3628909c
md"## Exercise 2 - _Square root by Newton's method_

Computing the square of a number is easy -- you already did it.

But how does one compute the square root of a number?

##### Algorithm:

Given: $x$

Output: $\sqrt{x}$

1. Take a guess `a`
1. Divide `x` by `a`
1. Set a = the average of `x/a` and `a`. (The square root must be between these two numbers. Why?)
1. Repeat until `x/a` is roughly equal to `a`. Return `a` as the square root.

In general, you will never get to the point where `x/a` is _exactly_ equal to `a`. So if our algorithm keeps going until `x/a == a`, then it will get stuck.

So instead, the algorithm takes a parameter `error_margin`, which is used to decide when `x/a` and `a` are close enough to halt.
"

# ╔═╡ 56866718-e6ce-11ea-0804-d108af4e5653
md"### (Optional) Exercise 2.1

Step 3 in the algorithm sets the new guess to be the average of `x/a` and the old guess `a`.

This is because the square root must be between the numbers `x/a` and `a`. Why?
"

# ╔═╡ bccf0e88-e754-11ea-3ab8-0170c2d44628
ex_1_1 = md"""
The sqrt(x) must always be between x/a and a (inclusive, since a=sqrt(x) is possible on the first guess) because they share an over-underestimate relationship, where one implies the other
"""

# you might need to wait until all other cells in this notebook have completed running.
# scroll down the page to see what's up
function newton_sqrt(a,x)
	
end

# ╔═╡ e7abd366-e7a6-11ea-30d7-1b6194614d0a
if !(@isdefined ex_1_1)
	md"""Do not change the name of the variable - write you answer as `ex_1_1 = "..."`"""
end

# ╔═╡ d62f223c-e754-11ea-2470-e72a605a9d7e
md"### Exercise 2.2

Write a function newton_sqrt(x) which implements the above algorithm."

# ╔═╡ 4896bf0c-e754-11ea-19dc-1380bb356ab6
function newton_sqrt(x, error_margin = 0.01, a = x / 2) # a=x/2 is the default value of `a`
	return x # this is wrong, write your code here!
end

# ╔═╡ 7a01a508-e78a-11ea-11da-999d38785348
newton_sqrt(2)

# ╔═╡ 682db9f8-e7b1-11ea-3949-6b683ca8b47b
let
	result = newton_sqrt(2, 0.01)
	if !(result isa Number)
		md"""
!!! warning "Not a number"
    `newton_sqrt` did not return a number. Did you forget to write `return`?
		"""
	elseif abs(result - sqrt(2)) < 0.01
		md"""
!!! correct
    Well done!
		"""
	else
		md"""
!!! warning "Incorrect"
    Keep working on it!
		"""
	end
end

# ╔═╡ 088cc652-e7a8-11ea-0ca7-f744f6f3afdd
md"""
!!! hint
    `abs(r - s)` is the distance between `r` and `s`
"""

# ╔═╡ c18dce7a-e7a7-11ea-0a1a-f944d46754e5
md"""
!!! hint
    If you're stuck, feel free to cheat, this is homework 0 after all 🙃

    Julia has a function called `sqrt`
"""

# ╔═╡ 9e777897-2dcb-42c8-8249-da02e68cb84f
md"## Exercise 3 - Getting started with `git` and GitHub

Create a [GitHub](https://github.com) account to be prepared for the introduction
to `git` in the next lecture.
"

# ╔═╡ 5e24d95c-e6ce-11ea-24be-bb19e1e14657
md"## (Optional) Exercise 4 - _Sierpinksi's triangle_

Sierpinski's triangle is defined _recursively_:

- Sierpinski's triangle of complexity N is a figure in the form of a triangle which is made of 3 triangular figures which are themselves Sierpinski's triangles of complexity N-1.

- A Sierpinski's triangle of complexity 0 is a simple solid equilateral triangle
"

# ╔═╡ 6b8883f6-e7b3-11ea-155e-6f62117e123b
md"To draw Sierpinski's triangle, we are going to use an external package, [_Compose.jl_](https://giovineitalia.github.io/Compose.jl/latest/tutorial). Let's import it!

A package contains a coherent set of functionality that you can often use a black box according to its specification. There are [lots of Julia packages](https://juliahub.com/ui/Home).
"

# ╔═╡ dbc4da6a-e7b4-11ea-3b70-6f2abfcab992
md"Just like the definition above, our `sierpinksi` function is _recursive_: it calls itself."

# ╔═╡ 02b9c9d6-e752-11ea-0f32-91b7b6481684
complexity = 3

# ╔═╡ 1eb79812-e7b5-11ea-1c10-63b24803dd8a
if complexity == 3
	md"""
Try changing the value of **`complexity` to `5`** in the cell above.

Hit `Shift+Enter` to affect the change.
	"""
else
	md"""
**Great!** As you can see, all the cells in this notebook are linked together by the variables they define and use. Just like a spreadsheet!
	"""
end

# ╔═╡ d7e8202c-e7b5-11ea-30d3-adcd6867d5f5
md"### Exercise 4.1

As you can see, the total area covered by triangles is lower when the complexity is higher."

# ╔═╡ f22222b4-e7b5-11ea-0ea0-8fa368d2a014
md"""
Can you write a function that computes the _area of `sierpinski(n)`_, as a fraction of the area of `sierpinski(0)`?

So:
```
area_sierpinski(0) = 1.0
area_sierpinski(1) = 0.??
...
```
"""

# ╔═╡ ca8d2f72-e7b6-11ea-1893-f1e6d0a20dc7
function area_sierpinski(n)
	return (3/4)^n
end

# ╔═╡ 71c78614-e7bc-11ea-0959-c7a91a10d481
if area_sierpinski(0) == 1.0 && area_sierpinski(1) == 3 / 4
	md"""
!!! correct
    Well done!
	"""
else
	md"""
!!! warning "Incorrect"
    Keep working on it!
	"""
end

# ╔═╡ c21096c0-e856-11ea-3dc5-a5b0cbf29335
md"**Let's try it out below:**"

# ╔═╡ 52533e00-e856-11ea-08a7-25e556fb1127
md"Complexity = $(@bind n Slider(0:6, show_value=true))"

# ╔═╡ c1ecad86-e7bc-11ea-1201-23ee380181a1
md"""
!!! hint
    Can you write `area_sierpinksi(n)` as a function of `area_sierpinski(n-1)`?
"""

# ╔═╡ a60a492a-e7bc-11ea-0f0b-75d81ce46a01
md"That's it for now, see you next week!"

# ╔═╡ dfdeab34-e751-11ea-0f90-2fa9bbdccb1e
triangle() = compose(context(), polygon([(1, 1), (0, 1), (1 / 2, 0)]))

# ╔═╡ b923d394-e750-11ea-1971-595e09ab35b5
# It does not matter which order you define the building blocks (functions) of the
# program in. The best way to organize code is the one that promotes understanding.

function place_in_3_corners(t)
	# Uses the Compose library to place 3 copies of t
	# in the 3 corners of a triangle.
	# treat this function as a black box,
	# or learn how it works from the Compose documentation here https://giovineitalia.github.io/Compose.jl/latest/tutorial/#Compose-is-declarative-1
	compose(context(),
			(context(1 / 4,   0, 1 / 2, 1 / 2), t),
			(context(0, 1 / 2, 1 / 2, 1 / 2), t),
			(context(1 / 2, 1 / 2, 1 / 2, 1 / 2), t))
end

# ╔═╡ e2848b9a-e703-11ea-24f9-b9131434a84b
function sierpinski(n)
	if n == 0
		triangle()
	else
		t = sierpinski(n - 1) # recursively construct a smaller sierpinski's triangle
		place_in_3_corners(t) # place it in the 3 corners of a triangle
	end
end

# ╔═╡ 9664ac52-e750-11ea-171c-e7d57741a68c
sierpinski(complexity)

# ╔═╡ df0a4068-e7b2-11ea-2475-81b237d492b3
sierpinski.(0:6)

# ╔═╡ 147ed7b0-e856-11ea-0d0e-7ff0d527e352
md"""

Sierpinski's triangle of complexity $(n)

 $(sierpinski(n))

has area **$(area_sierpinski(n))**

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Compose = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Compose = "~0.9.7"
PlutoTeachingTools = "~0.2.15"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.1"
manifest_format = "2.1"
project_hash = "bb4fccc24c43367a667f5ef1cfa3fb1bba8546d5"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
registries = "General"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "61761f58648aa7217445f24f841839b78c712232"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.3"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "291665b547f137df070e4dd83e432b5fee8cc4a0"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.2"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
registries = "General"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "d75431a71f82758e218779ccb3369f3243fd2bc1"
registries = "General"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.7"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random"]
git-tree-sha1 = "210525997a2434caa497655f86d151b1bcad0dce"
registries = "General"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.12.0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
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
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.18.0+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.1+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "497f602dc6d4d894e000ad6ee13b6dfd3c0be94a"
registries = "General"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.10.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "f9b03759e9ef463718934fbed30820b39997511e"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "ba0dc8a8a67cacac4842631f960c046e4e563675"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.8"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
registries = "General"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "aea4eede5ab3ee188906d0cf3bbfa36eb543dccc"
registries = "General"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.8"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "5d9ab1a4faf25a62bb9d07ef0003396ac258ef1c"
registries = "General"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.15"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "5005266de4bfe50e53ff44a5cb5c540b6e47a254"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CRC32c", "CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "177e0fc0e89b18c567e90d04f841a8304b22f097"
registries = "General"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.17.2"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.67.1+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─ee2e9533-b0f7-4985-9f15-edbbb52b6046
# ╟─a2181260-e6cd-11ea-2a69-8d9d31d1ef0e
# ╠═b572d2c8-556d-4520-a8d9-ce0b49ea3b8c
# ╠═5ba2e5a9-4107-4af8-8cb9-6cd4dd2f0a96
# ╟─a05d2bc8-7024-11eb-08cb-196543bbb8fd
# ╠═e02f7ea6-7024-11eb-3672-fd59a6cff79b
# ╟─6acef56c-7025-11eb-2524-819c30a75d39
# ╟─348cea34-7025-11eb-3def-41bbc16c7512
# ╟─ea716adf-7782-44b2-a330-0424c820c713
# ╟─154d5ac6-9815-4c32-bd6b-631042a4c5af
# ╟─ac07d216-a2be-4761-80a4-e43a95a4c73a
# ╠═d5e8eba8-4d4f-4f18-b8ac-dc55fc4643ce
# ╟─ad116d8e-9286-40f8-9c7f-b0be6d3cdea7
# ╟─c2d9ba9e-e9a5-400e-8a47-92415d08fef2
# ╟─6a19dc43-7cf3-40a9-9275-5434d75942ff
# ╠═a9b3bef1-f400-4352-b02c-743caf51335f
# ╠═32c0ed9d-0248-4812-9873-81cba8efb621
# ╟─bcc0bead-bde4-4b72-80ba-9803b70f517b
# ╟─b64a7a85-7855-4448-9906-1f4a64d497b6
# ╠═9ba4b355-1e61-47b9-ac45-ade50dd017c3
# ╟─e4fe747c-26e4-4b11-b14e-fab4e2d536ea
# ╟─d4de36a4-caff-4a68-a3ff-a49f3cc446ce
# ╠═1bd97b90-6667-4be8-9e1a-8a4623ca74d2
# ╟─bdf1bdc5-d3e8-41de-9416-1c31e0d0965a
# ╟─a18a6c6c-3c70-4655-bcf6-e0d1a6119fbd
# ╟─cf83ef0c-665b-4f1e-bbf3-8ef04c6ab2ac
# ╠═f7240706-0bb8-4740-bdf4-d796b9eb647d
# ╟─828aab16-921a-447b-b14e-6c5a2cfffdff
# ╟─7b85f2c8-0cab-4382-9ee6-f8bcd3bc9da8
# ╠═a90ea392-291b-4357-9b2f-7b450a6eb19d
# ╠═d74266d2-1d71-4a71-af13-b0c07bb73343
# ╟─53ce3ab7-3edd-4a57-920b-84c382bc8de7
# ╟─b2705e7f-61d8-421b-bda5-286cf0fb879d
# ╠═c98e7e36-cbc7-4235-8c27-43a5fc5fddca
# ╠═c12b18d2-76dd-4ea1-a289-8f20ff791f1b
# ╟─d6f38ac5-e88e-44cd-be62-2400927fe7bc
# ╟─339c2d5c-e6ce-11ea-32f9-714b3628909c
# ╟─56866718-e6ce-11ea-0804-d108af4e5653
# ╠═bccf0e88-e754-11ea-3ab8-0170c2d44628
# ╟─e7abd366-e7a6-11ea-30d7-1b6194614d0a
# ╟─d62f223c-e754-11ea-2470-e72a605a9d7e
# ╟─4896bf0c-e754-11ea-19dc-1380bb356ab6
# ╠═7a01a508-e78a-11ea-11da-999d38785348
# ╟─682db9f8-e7b1-11ea-3949-6b683ca8b47b
# ╟─088cc652-e7a8-11ea-0ca7-f744f6f3afdd
# ╟─c18dce7a-e7a7-11ea-0a1a-f944d46754e5
# ╟─9e777897-2dcb-42c8-8249-da02e68cb84f
# ╟─5e24d95c-e6ce-11ea-24be-bb19e1e14657
# ╟─6b8883f6-e7b3-11ea-155e-6f62117e123b
# ╠═d6ee91ea-e750-11ea-1260-31ebf3ec6a9b
# ╟─dbc4da6a-e7b4-11ea-3b70-6f2abfcab992
# ╠═e2848b9a-e703-11ea-24f9-b9131434a84b
# ╠═9664ac52-e750-11ea-171c-e7d57741a68c
# ╠═02b9c9d6-e752-11ea-0f32-91b7b6481684
# ╟─1eb79812-e7b5-11ea-1c10-63b24803dd8a
# ╟─d7e8202c-e7b5-11ea-30d3-adcd6867d5f5
# ╠═df0a4068-e7b2-11ea-2475-81b237d492b3
# ╟─f22222b4-e7b5-11ea-0ea0-8fa368d2a014
# ╠═ca8d2f72-e7b6-11ea-1893-f1e6d0a20dc7
# ╟─71c78614-e7bc-11ea-0959-c7a91a10d481
# ╟─c21096c0-e856-11ea-3dc5-a5b0cbf29335
# ╟─52533e00-e856-11ea-08a7-25e556fb1127
# ╟─147ed7b0-e856-11ea-0d0e-7ff0d527e352
# ╟─c1ecad86-e7bc-11ea-1201-23ee380181a1
# ╟─a60a492a-e7bc-11ea-0f0b-75d81ce46a01
# ╟─dfdeab34-e751-11ea-0f90-2fa9bbdccb1e
# ╟─b923d394-e750-11ea-1971-595e09ab35b5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
