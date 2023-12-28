model_data_file(power) = datadir(
	"qam_data",
	"model_power_$(power)_wo_noise.parquet"
)

signal_data_file(power::Int,with_noise::Bool,pilots::Int) = datadir(
	"qam_data",
	"points_power_$(power)_pilots_$(pilots)_$(with_noise ? "w" : "wo")_noise.parquet"
)

noise_data_file() = datadir("noise_info.parquet")

encoding_data_file() = datadir("constellation_info.parquet")

dataset(fn) = Parquet2.Dataset(fn)|>DataFrame

Base.@kwdef struct AlgorithmParams
    T::Int
    step::Int
    w::Float64=0.25
    niter::Int=10
    showprogressinfo::Bool=true
end

AlgorithmParams(d) = AlgorithmParams(;d...)

Base.iterate(a::AlgorithmParams,args...) = iterate(struct2dict(a),args...)
DrWatson.allaccess(::AlgorithmParams) = (:T,:step,:w,:niter)
DrWatson.default_allowed(::AlgorithmParams) = (Real,)

Base.@kwdef struct DecodingTask
    power::Int
    with_noise::Bool
    pilots::Int
    k::Int
    pilots_period::Int=100
    skip_batches::Int=0
    load_batches::Int=1_000_000
    is_simulation::Bool=false
    alg_params::AlgorithmParams=AlgorithmParams(T=10,step=8)
    pilot_point::Tuple{Int,Int}=(1,1)
    qam_encoding::Dict{Tuple{Int,Int},Int}=DataPrep.get_qam_encoding(dataset(encoding_data_file()))
    noise_info::NamedTuple=get_noise_info(dataset(noise_data_file()),power,with_noise)
    model_file::String=model_data_file(power)
    signal_file::String=signal_data_file(power,with_noise,pilots)
end

DecodingTask(d) = DecodingTask(;d...)

DrWatson.default_expand(::DecodingTask) = ["alg_params"]
DrWatson.default_allowed(::DecodingTask) = (Real,Bool,AlgorithmParams)
DrWatson.allaccess(::DecodingTask) = (:power,:with_noise,:pilots,:pilots_period,:k,:skip_batches,:load_batches,:alg_params)
DrWatson.default_prefix(t::DecodingTask) = t.is_simulation ? "simulation-decoded" : "points-decoded"

function decodingtask2dict(t::DecodingTask)
    d = struct2dict(t)
    d[:alg_params] = struct2dict(t.alg_params)
    return d
end

function decoding_tasks_list(; alg_params=nothing,kw...)
    d = Dict(kw)
    if !isnothing(alg_params)
        d[:alg_params] = alg_params|>pairs|>Dict|>dict_list.|>AlgorithmParams
    end
    return d|>dict_list.|>DecodingTask
end