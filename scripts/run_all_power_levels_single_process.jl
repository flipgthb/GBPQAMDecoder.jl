using DrWatson
@quickactivate "GBPQAMDecoder"

using Chain, DataFrames, Dates, LinearAlgebra, Parquet2, ProgressMeter, StatsBase, Transducers
using DataPrep
using GBPAlgorithm

include(srcdir("data_files_utilities.jl"))

function solve_decoding_task(task_data,n,T,w,niter; showprogressinfo=true)
    N = length(task_data.qam_encoding)
    M = ones(N,N,N,T)
    P = ones(N,T)
    b = copy(P)
    p = Progress(n)
    withprogress(Tables.rowtable(task_data.signal_table); interval=10^-2)|>
        Partition(T)|>
        Map() do x
            y = DataFrame(x)
            DataPrep.memory_factor!(M,task_data.model_table,y.Rx)
            GBPAlgorithm.uniform_prior!(P)
            (;Rs) = GBPAlgorithm.decode!(b,M,P,w)|>Drop(niter-1)|>Take(1)|>collect|>only
            if showprogressinfo
                next!(p)
            end
            transform!(y,:Ts=>(x->Rs)=>:Rs)
            transform!(y,[:Ts,:Rs]=>((t,r)->t.!=r)=>:error)
        end|>
        Take(n)|>
        foldxl(vcat)
end

function info_msg(task_info,T,n)
    (;power,with_noise,with_pilotwave,number_of_mixture_components) = task_info
    @info "Running algorithm:" "Avg. Power [dBm]"=power "With 4.5dB noise"=with_noise "With pilot wave"=with_pilotwave "Number of mixture components"=number_of_mixture_components "Sequence length"=T "Number of sequences"=n
end

function summarize_results(res)
    mapreduce(vcat,res) do r
        (;task_info,results,number_of_sequences,sequence_length) = r
        (;power,with_noise,with_pilotwave,number_of_mixture_components) = task_info
        ber_gbp=(results.error./4)|>mean
        (;power,with_noise,with_pilotwave,number_of_mixture_components,sequence_length,number_of_sequences,ber_gbp)
    end|>DataFrame
end

# const number_of_mixture_components = 2
# const decoding_tasks = (
#     decoding_task_info(;power,with_noise,with_pilotwave=true,number_of_mixture_components)
#     for power in -2:8
#     for with_noise in false:true
# )

@info "Warming up..."
let n=3, T=5, w=0.25, niter=10, k=2

    decoding_tasks = (
        decoding_task_info(;power,with_noise,with_pilotwave=true,number_of_mixture_components=k)
        for power in -1:1
        for with_noise in false:true
    )

    map(decoding_tasks) do task_info
        @chain task_info begin
            load_problem_data
            solve_decoding_task(n,T,w,niter;showprogressinfo=false)
            (;task_info,results=_)
        end
    end

    map(decoding_tasks) do task_info
        @chain task_info begin
            load_problem_data
            solve_decoding_task(n,T,w,niter;showprogressinfo=false)
            (;task_info,results=_)
        end
    end
end

println("")

# alg_params = (;n=100, T=5, w=0.25, niter=10, k=number_of_mixture_components,date=today())
function run_it(;n,T,w,niter,k,savedir="decoding_results")
    # number_of_mixture_components = 2
    decoding_tasks = (
        decoding_task_info(;power,with_noise,with_pilotwave=true,number_of_mixture_components=k)
        for power in -2:8
        for with_noise in false:true
    )
    alg_params = (;n, T, w, niter,k,data=today())

    @info "Running GBP decoder for all powers"
    t0 = now()
    res = mapreduce(vcat,decoding_tasks) do task_info
        info_msg(task_info,T,n)
        @chain task_info begin
            load_problem_data
            solve_decoding_task(n,T,w,niter)
            (;task_info,number_of_sequences=n,sequence_length=T,results=_)
        end
    end
    tf = now()
    @info "Done." "Elapsed time"=canonicalize(tf-t0)

    dir = datadir("exp_pro",savedir)
    @info "Saving summary..." "Directory"=dir
    mkpath(dir)
    @assert ispath(dir)
    fn = joinpath(dir,savename("results",alg_params,"parquet";equals="_",connector="-"))
    d = summarize_results(res)|>
        savedata(fn)

    println("\n")
    (;summary=d,results=res)
end

run_it(;n=2500, T=5, w=0.25, niter=10, k=2,savedir="decoding_results")
run_it(;n=2500, T=5, w=0.25, niter=10, k=1,savedir="decoding_results")
run_it(;n=2500, T=5, w=0.25, niter=10, k=3,savedir="decoding_results")
run_it(;n=2500, T=10, w=0.25, niter=10, k=2,savedir="decoding_results")
run_it(;n=2500, T=50, w=0.25, niter=10, k=2,savedir="decoding_results");