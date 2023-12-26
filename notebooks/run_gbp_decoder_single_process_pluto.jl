### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 90f835b0-a3e8-11ee-3e33-9719b5d2efcb
using DrWatson

# ╔═╡ 0472fffd-4686-43af-8bd9-9fd96ec05efc
@quickactivate "GBPQAMDecoder"

# ╔═╡ f6d84f50-b548-4b75-ae2d-ea43c7fc888d
using Chain,
    DataFrames,
    Dates,
    LinearAlgebra,
    Parquet2,
    ProgressMeter,
    Random,
    StatsBase,
    Transducers

# ╔═╡ cb162620-67e9-4124-8735-6c1300fabb55
using DataPrep

# ╔═╡ aae0e074-a287-42a4-bc6d-a6d9637aad0b
using GBPAlgorithm

# ╔═╡ d42b9d61-0ee7-4907-bae7-b785a6f13a99
using PlutoUI

# ╔═╡ 9acb786d-8e76-44ee-b817-f0430405dc2a
include(srcdir("data_utilities.jl"))

# ╔═╡ af1edd4f-976c-4a43-8d76-d11c0695fa8e
include(srcdir("decoding.jl"))

# ╔═╡ 3efb8183-f539-4de5-b792-8b81750c573e
warm_up_results = let
    run_it(;
        power_vals=0:1,with_noise_vals=true:true,pilots_vals=1:1,k_vals=2:2,
        T=10,w=0.25,niter=10,pilots_period=100,step=8,is_simulation=false,
        load_take=200,load_skip=0,should_save=false,savedir="test_$(today())",
        start_msg="Warmming up...", showprogressinfo=false
    );

    run_it(;
        power_vals=0:1,with_noise_vals=true:true,pilots_vals=1:1,k_vals=2:2,
        T=10,w=0.25,niter=10,pilots_period=100,step=8,is_simulation=false,
        load_take=200,load_skip=0,should_save=true,savedir="test_$(today())",
        start_msg="one more, this time saving...", showprogressinfo=false
    );
end;

# ╔═╡ 7c865aca-8306-40a2-a35e-afd627c12c1d
function get_available_data_parameters()

	is_model_filename(fn) = occursin(r"model",fn)
	get_power_from_filename(fn) = parse(Int,match(r"power_(?<p>-?\d+)",fn)["p"])
	get_with_noise_from_filename(fn) = match(r"(?<n>w|wo)_noise",fn)["n"] == "w"
	get_pilots_from_filename(fn) = parse(Int,match(r"pilots_(?<w>\d+)",fn)["w"])

	function get_available_model_power_levels()
		@chain datadir("qam_data") begin
			readdir
			filter(is_model_filename,_)
			map(get_power_from_filename,_)
		end
	end

	is_signal_filename(fn) = occursin(r"point",fn)
	valid_power_level_filename(fn) = get_power_from_filename(fn) in get_available_model_power_levels()
	
	@chain datadir("qam_data") begin
		readdir
		filter(_) do fn
			is_signal_filename(fn) && valid_power_level_filename(fn)
		end
		map(_) do fn
			power = get_power_from_filename(fn)
			with_noise = get_with_noise_from_filename(fn)
			pilots = get_pilots_from_filename(fn)
			(;power,with_noise,pilots)
		end
	end
end

# ╔═╡ f7e0d5e4-0fa4-4d03-a1d8-7ab640109f0f
function parameters_ui()
	params = get_available_data_parameters()|>
				DataFrame

	power = params.power|>unique|>sort!
	(min_power,max_power) = power|>extrema

	pilots = params.pilots|>unique|>sort!
	min_pilots = pilots|>minimum

	return PlutoUI.combine() do Child
		inputs = [
			md"""Avg. Power [dBm]: $(
					Child("power_vals", RangeSlider(min_power:max_power;default=0:0,show_value=true))
			)
			""",
			md"""Noise condition: $(
					Child("with_noise_vals", 
						MultiCheckBox([false=>"Without",true=>"With 4.4dB noise"]; default=[false])
					)
			)
			""",
			md"""Pilots: $(
					Child("pilots_vals", MultiCheckBox(pilots; default=[min_pilots]))
			)
			""",
			md"""Number of mixture components: $(
					Child("k_vals", MultiCheckBox(1:5; default=[2]))
			)
			""",
			md"""Rolling window length: $(
					Child("T", Slider(4:15; default=10, show_value=true))
			)
			""",
			md"""Rolling window step: $(
					Child("step", Slider(1:15; default=8, show_value=true))
			)
			""",
			md"""Pilots period: $(
					Child("pilots_period",Select([100]; default=[100]))
			)
			""",
			md"""Gradient descent step size: $(
					Child("w", Slider(0:0.05:1.0; default=0.25, show_value=true))
			)
			""",
			md"""Number of GBP iterations: $(
					Child("niter", Slider(1:20; default=10, show_value=true))
			)
			""",
			md"""Simulate signal: $(
					Child("is_simulation", CheckBox())
			)
			""",
			md""" Drop $(
					Child("load_skip",NumberField(0:1_000_000; default=0))
			) symbols from signal data file (ignored if is simulation)
			""",
			md""" Load $(
					Child("load_take",NumberField(0:1_000_000; default=200))
			) symbols from signal data file (used as number of simulated symbols)
			""",
			md"""Save results: $(
					Child("should_save", CheckBox(;default=false))
			)
			""",
			md"""
			Save directory (under $(datadir("results"))): $(Child("savedir",TextField()))
			""",
			md"""Show progress: $(Child("showprogressinfo",CheckBox(;default=true)))
			""",
			md"""Start message: $(
					Child("start_msg",TextField((50,1);default="Running GBP decoder..."))
			)
			"""
		]

		md"""### Setup decoding tasks parameters
		$(inputs)
		"""
	end
end

# ╔═╡ bf8a6bb2-6648-4daa-a96c-7289995b4b3e
md"## Running the GBP decoding algorithm"

# ╔═╡ f80e34a1-2f3d-4cfc-a689-ebd1e2cc3574
@bind params confirm(parameters_ui())

# ╔═╡ fa588b2b-dcc5-4b75-87e0-490aecff8016
md"### Run decoder"

# ╔═╡ 668740f7-776a-4bff-b4fe-f5c023e4b6b6
run_results = run_it(; params...);

# ╔═╡ Cell order:
# ╠═90f835b0-a3e8-11ee-3e33-9719b5d2efcb
# ╠═0472fffd-4686-43af-8bd9-9fd96ec05efc
# ╠═f6d84f50-b548-4b75-ae2d-ea43c7fc888d
# ╠═cb162620-67e9-4124-8735-6c1300fabb55
# ╠═aae0e074-a287-42a4-bc6d-a6d9637aad0b
# ╠═d42b9d61-0ee7-4907-bae7-b785a6f13a99
# ╠═9acb786d-8e76-44ee-b817-f0430405dc2a
# ╠═af1edd4f-976c-4a43-8d76-d11c0695fa8e
# ╟─3efb8183-f539-4de5-b792-8b81750c573e
# ╟─7c865aca-8306-40a2-a35e-afd627c12c1d
# ╟─f7e0d5e4-0fa4-4d03-a1d8-7ab640109f0f
# ╟─bf8a6bb2-6648-4daa-a96c-7289995b4b3e
# ╠═f80e34a1-2f3d-4cfc-a689-ebd1e2cc3574
# ╟─fa588b2b-dcc5-4b75-87e0-490aecff8016
# ╠═668740f7-776a-4bff-b4fe-f5c023e4b6b6
