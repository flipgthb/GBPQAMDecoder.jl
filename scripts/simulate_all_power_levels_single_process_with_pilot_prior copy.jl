using DrWatson
@quickactivate "GBPQAMDecoder"

using Chain, DataFrames, Dates, LinearAlgebra, Parquet2, ProgressMeter, StatsBase, Transducers
using DataPrep
using GBPAlgorithm

include(srcdir("data_files_utilities.jl"))
include(srcdir("transmission_simulator.jl"))

function add_pilots_info(task_data,pilots,pilots_period,pilot_point)
    pilot_symbol = task_data.qam_encoding[pilot_point]
    @chain task_data.signal_table begin
        transform!(eachindex=>:t)
        transform!(:t=>ByRow(t->mod(t,1:pilots_period)<=pilots)=>:is_pilot)
        transform!(:is_pilot=>ByRow(w->w ? pilot_symbol : Colon())=>:pilot_index)
    end
    task_data
end

function solve_decoding_task(task_data,n,T,w,niter;showprogressinfo=true)
    N = length(task_data.qam_encoding)
    M = ones(N,N,N,T)
    P = ones(N,T)
    b = copy(P)
    p = Progress(n)
    step = task_data.task_info.pilots_period
    withprogress(Tables.rowtable(task_data.signal_table); interval=10^-2)|>
        Partition(T,step)|>
        Map() do x
            y = DataFrame(x)
            DataPrep.memory_factor!(M,task_data.model_table,y.Rx)
            GBPAlgorithm.pilot_prior!(P,y.pilot_index)
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
    (;power,with_noise,pilots,number_of_mixture_components) = task_info
    @info "Running algorithm:" "Avg. Power [dBm]"=power "With 4.5dB noise"=with_noise "Pilots"=pilots "Number of mixture components"=number_of_mixture_components "Sequence length"=T "Number of sequences"=n
end

function summarize_results(res)
    mapreduce(vcat,res) do r
        (;task_info,results,number_of_sequences,sequence_length) = r
        (;power,with_noise,number_of_mixture_components) = task_info
        (;pilots,pilots_period,symbol_sequence_length) = task_info        
        # ber_gbp=(filgresults.error./4)|>mean
        ber_gbp=combine(filter(row->!row.is_pilot,results),:error=>(x->mean(x)/4)=>:ber_gbp).ber_gbp
        ber_pilots=combine(filter(row->row.is_pilot,results),:error=>(x->mean(x)/4)=>:ber_pilots).ber_pilots
        (;power,with_noise,pilots,number_of_mixture_components,sequence_length,number_of_sequences,ber_gbp,ber_pilots)
    end|>DataFrame
end

@info "Warming up..."
let n=3, T=5, w=0.25, niter=10, k=2, pilots=3, pilots_period=100, symbol_sequence_length=n*T
    decoding_tasks = (
        simulated_decoding_task_info(;
            power,with_noise,pilots,pilots_period,
            symbol_sequence_length,
            number_of_mixture_components=k
        )
        for power in -1:1
        for with_noise in false:true
    )

    map(decoding_tasks) do task_info
        (;pilots,pilots_period,pilot_point) = task_info
        @chain task_info begin
            load_problem_data_for_simulation
            add_pilots_info(pilots,pilots_period,pilot_point)
            solve_decoding_task(n,T,w,niter;showprogressinfo=false)
            (;task_info,results=_)
        end
    end

    map(decoding_tasks) do task_info
        (;pilots,pilots_period,pilot_point) = task_info
        @chain task_info begin
            load_problem_data_for_simulation
            add_pilots_info(pilots,pilots_period,pilot_point)
            solve_decoding_task(n,T,w,niter;showprogressinfo=false)
            (;task_info,results=_)
        end
    end|>summarize_results
end

println("")

# add_noise_var(task_info) = (;task_info...,noise_var=sqrt(task_info.noise_sigma))

function run_it(;n,T,w,niter,k,pilots=2,pilots_period=8,symbol_sequence_length=T+(n-1)*pilots_period,savedir="")
    decoding_tasks = (
        simulated_decoding_task_info(;
            power,with_noise,pilots,pilots_period,
            symbol_sequence_length,
            number_of_mixture_components=k
        )#|>add_noise_var
        for power in -2:8
        for with_noise in false:true
    )
    alg_params = (;n, T, w, niter,k,pilots,date=today())

    @info "Running GBP decoder for all powers for a simulated sequence"
    t0 = now()
    res = mapreduce(vcat,decoding_tasks) do task_info
        info_msg(task_info,T,n)
        (;pilots,pilots_period,pilot_point) = task_info
        @chain task_info begin
            load_problem_data_for_simulation
            add_pilots_info(pilots,pilots_period,pilot_point)
            solve_decoding_task(n,T,w,niter)
            (;task_info,number_of_sequences=n,sequence_length=T,results=_)
        end
    end
    tf = now()
    @info "Done." "Elapsed time"=canonicalize(tf-t0)

    dir = datadir("results","simulations",savedir)
    @info "Saving summary..." "Directory"=dir
    mkpath(dir)
    @assert ispath(dir)
    fn = joinpath(dir,savename("results",alg_params,"parquet";equals="_",connector="-"))
    d = summarize_results(res)|>
        savedata(fn)

    println("\n")
    (;summary=d,results=res)
end

# function simple_cases(;savedir)
#     run_it(;n=2500,  T=5, w=0.25, niter=10, k=2, pilots=0, savedir)
#     run_it(;n=2500, T=10, w=0.25, niter=10, k=2, pilots=0, savedir)
#     return
# end

println("""
Run `run_it(; n::Int, T::Int, w::Float64, niter::Int, k::Int, pilots::Int[, savedir::String])`
to run the GBP decoding algorithm for all available power levels and noise conditions given:
    - `n` sequences
    - each sequence with length `T`
    - using a mixture with `k` Gaussian components                (k in {1,2,3,4,5})
    - using a gradient descent step of `w`                        (w in 0..1) 
    - on signal generated with `pilots` waves every 100 symbols   (pilots in {0,1,2,3})
    - and optionally, saves it to a subdirectory `datadir("results",savedir)`

Run `simple_cases(;savedir)` to run the GBP algorithm for a couple of selected cases.
""")