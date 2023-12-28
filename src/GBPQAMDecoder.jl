module GBPQAMDecoder

using DrWatson
@quickactivate "GBPQAMDecoder"

export solve_decoding_task, load_problem_data, decoding_tasks_list, DecodingTask, AlgorithmParams

using Chain,
    DataFrames,
    Dates,
    LinearAlgebra,
    Parquet2,
    ProgressMeter,
    Random,
    StatsBase,
    Transducers

using DataPrep     # local module
using GBPAlgorithm # local module

include(srcdir("decoding_tasks.jl"))
include(srcdir("signal_simulator.jl"))
include(srcdir("decoding.jl"))

# model_data_file(power) = datadir(
# 	"qam_data",
# 	"model_power_$(power)_wo_noise.parquet"
# )

# signal_data_file(power::Int,with_noise::Bool,pilots::Int) = datadir(
# 	"qam_data",
# 	"points_power_$(power)_pilots_$(pilots)_$(with_noise ? "w" : "wo")_noise.parquet"
# )

# noise_data_file() = datadir("noise_info.parquet")

# encoding_data_file() = datadir("constellation_info.parquet")

# dataset(fn) = Parquet2.Dataset(fn)|>DataFrame

# Base.@kwdef struct AlgorithmParams
#     T::Int
#     step::Int
#     w::Float64=0.25
#     niter::Int=10
#     showprogressinfo::Bool=true
# end

# AlgorithmParams(d) = AlgorithmParams(;d...)

# Base.iterate(a::AlgorithmParams,args...) = iterate(struct2dict(a),args...)
# DrWatson.allaccess(::AlgorithmParams) = (:T,:step,:w,:niter)
# DrWatson.default_allowed(::AlgorithmParams) = (Real,)

# Base.@kwdef struct DecodingTask
#     power::Int
#     with_noise::Bool
#     pilots::Int
#     k::Int
#     pilots_period::Int=100
#     skip_batches::Int=0
#     load_batches::Int=1_000_000
#     is_simulation::Bool=false
#     alg_params::AlgorithmParams=AlgorithmParams(T=10,step=8)
#     pilot_point::Tuple{Int,Int}=(1,1)
#     qam_encoding::Dict{Tuple{Int,Int},Int}=DataPrep.get_qam_encoding(dataset(encoding_data_file()))
#     noise_info::NamedTuple=get_noise_info(dataset(noise_data_file()),power,with_noise)
#     model_file::String=model_data_file(power)
#     signal_file::String=signal_data_file(power,with_noise,pilots)
# end

# DecodingTask(d) = DecodingTask(;d...)

# DrWatson.default_expand(::DecodingTask) = ["alg_params"]
# DrWatson.default_allowed(::DecodingTask) = (Real,Bool,AlgorithmParams)
# DrWatson.allaccess(::DecodingTask) = (:power,:with_noise,:pilots,:pilots_period,:k,:skip_batches,:load_batches,:alg_params)
# DrWatson.default_prefix(t::DecodingTask) = t.is_simulation ? "simulation-decoded" : "points-decoded"

# function decodingtask2dict(t::DecodingTask)
#     d = struct2dict(t)
#     d[:alg_params] = struct2dict(t.alg_params)
#     return d
# end

# function decoding_tasks_list(; alg_params=nothing,kw...)
#     d = Dict(kw)
#     if !isnothing(alg_params)
#         d[:alg_params] = alg_params|>pairs|>Dict|>dict_list.|>AlgorithmParams
#     end
#     return d|>dict_list.|>DecodingTask
# end

# function add_pilots_info!(signal_table,task_info)
#     (;pilots,pilots_period) = task_info
#     @chain signal_table begin
#         transform!(eachindex=>:t)
#         transform!(:t=>ByRow(t->mod(t,1:pilots_period)<=pilots)=>:is_pilot)
#     end
# end

# function collapse_prior!(factors,sequence,T,step,part_idx)
#     overlap = part_idx > 1 ? pairs(last(factors.Rs,T-step)) : ()
#     pilots = (r.i=>r.Ts for r in eachrow(sequence) if r.is_pilot)
#     GBPAlgorithm.collapse_prior!(factors,overlap...,pilots...)
#     return factors
# end

# function load_problem_data(task_info)
#     (;k,qam_encoding) = task_info
#     (;noise_var) = task_info.noise_info
# 	model_table = @chain task_info.model_file begin
# 		dataset
# 		DataPrep.parse_model_data(k,noise_var,qam_encoding)
# 	end

#     (;load_batches,skip_batches,pilots_period) = task_info
# 	signal_table = if !task_info.is_simulation
# 			@chain task_info.signal_file begin
# 			Parquet2.Dataset
# 			Tables.rows
# 			_|>Drop(skip_batches*pilots_period)|>Take(load_batches*pilots_period)|>DataFrame
# 			DataPrep.parse_signal_data(qam_encoding)
#             add_pilots_info!(task_info)
# 		end
# 	else
# 		@warn "Simulating signal..."
# 		symbol_sequence_length=min(load_batches*pilots_period,size(dataset(task_info.signal_file),1))
# 		add_pilots_info!(
#             simulate_signal_table(
#                 task_info,
#                 symbol_sequence_length,
#                 model_table,
#                 qam_encoding
#             ),
#             task_info
#         )
# 	end

#     return (;model_table,signal_table)
# end

# progress_info_msg(prog,showinfo) = showinfo && next!(prog)

# function task_info_msg(task_info)
#     (;T,step) = task_info.alg_params
#     (;power,with_noise,pilots,k) = task_info
#     @info "Running algorithm:" "Avg. Power [dBm]"=power "With 4.5dB noise"=with_noise "Pilots"=pilots "Number of mixture components"=k "Sequence length"=T "Step"=step
# end

# function save_info_msg(dir)
#     @info "Saved decoded points" "Directory"=dir
# end

# function get_noise_info(noise_data,power,with_noise)
#     @chain noise_data begin
#         subset(:pdbm=>ByRow(p->p==power))
#         map(eachrow(_)) do r
#             noise_sigma = with_noise ? r.sigma*r.scale : 0.0
#             noise_var = noise_sigma^2
#             (;noise_sigma,noise_var,noise_scale=r.scale)
#         end
#         only
#     end
# end

# getresults(sequence,step,part_idx) = part_idx > 1 ? last(sequence,step) : sequence

# maxparts(nsyms,partlen,step) = floor(Int,1 + (nsyms - partlen)/step)
# maxparts(seqdf::DataFrame,partlen,step) = maxparts(size(seqdf,1),partlen,step)

# function solve_decoding_task(task_info)
#     task_info_msg(task_info)
#     (;model_table,signal_table) = load_problem_data(task_info)
#     (;T,w,niter,step,showprogressinfo) = task_info.alg_params
#     N = length(task_info.qam_encoding)
#     decode_iter! = GBPAlgorithm.GBPDecoder(N,T,w)
#     factors = GBPAlgorithm.Factors(N,T)
#     n = maxparts(size(signal_table,1),T,step)
#     prog = Progress(n; showspeed=true)
#     R = withprogress(eachrow(signal_table); interval=10^-2)|>
#         Partition(T,step)|>
#         Enumerate()|>
#         Map() do (part_idx,part)
#             sequence = DataFrame(part)
#             transform!(sequence,eachindex=>:i)
#             GBPAlgorithm.reset_msg!(decode_iter!)
#             DataPrep.memory_factor!(factors.M,model_table,sequence.Rx)
#             collapse_prior!(factors,sequence,T,step,part_idx)
#             decode_iter!(factors)|>Drop(niter-1)|>Take(1)|>collect|>only
#             GBPAlgorithm.beliefs!(factors,decode_iter!)
#             transform!(sequence,:Ts=>(x->copy(factors.Rs))=>:Rs)
#             transform!(sequence,[:Ts,:Rs]=>((t,r)->t.!=r)=>:error)
#             progress_info_msg(prog,showprogressinfo)
#             getresults(sequence,step,part_idx)
#         end|>
#         Take(n)|>
#         foldxl(vcat)
#     results = select!(R,
#                 :Tx=>ByRow(((x,y),)->(;Tx_x=x,Tx_y=y))=>AsTable,
#                 :Rx=>ByRow(((x,y),)->(;Rx_x=x,Rx_y=y))=>AsTable,
#                 :Ts,:Rs,:error,:is_pilot
#             )

#     decoding_task = decodingtask2dict(task_info)
#     return @strdict decoding_task results
# end

end # end Decoding module