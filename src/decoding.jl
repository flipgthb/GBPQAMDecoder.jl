progress_info_msg(prog,showinfo) = showinfo && next!(prog)

function task_info_msg(task_info)
    (;T,step) = task_info.alg_params
    (;power,with_noise,pilots,number_of_mixture_components) = task_info
    @info "Running algorithm:" "Avg. Power [dBm]"=power "With 4.5dB noise"=with_noise "Pilots"=pilots "Number of mixture components"=number_of_mixture_components "Sequence length"=T "Step"=step
end

function save_info_msg(dir)
    @info "Saved decoded points" "Directory"=dir
end

function add_pilots_info(task_data)
    (;pilots,pilots_period) = task_data.task_info
    @chain task_data.signal_table begin
        transform!(eachindex=>:t)
        transform!(:t=>ByRow(t->mod(t,1:pilots_period)<=pilots)=>:is_pilot)
    end
    task_data
end

function collapse_prior!(factors,sequence,T,step,part_idx)
    overlap = part_idx > 1 ? pairs(last(factors.Rs,T-step)) : ()
    pilots = (r.i=>r.Ts for r in eachrow(sequence) if r.is_pilot)
    GBPAlgorithm.collapse_prior!(factors,overlap...,pilots...)
    return factors
end

get_results(sequence,step,part_idx) = part_idx > 1 ? last(sequence,step) : sequence

maxparts(nsyms,partlen,step) = floor(Int,1 + (nsyms - partlen)/step)
maxparts(seqdf::DataFrame,partlen,step) = maxparts(size(seqdf,1),partlen,step)

function solve_decoding_task(task_data)
    (;T,w,niter,step,showprogressinfo) = task_data.task_info.alg_params
    N = length(task_data.qam_encoding)
    decode_iter! = GBPAlgorithm.GBPDecoder(N,T,w)
    factors = GBPAlgorithm.Factors(N,T)
    n = maxparts(size(task_data.signal_table,1),T,step)
    prog = Progress(n; showspeed=true)
    R = withprogress(eachrow(task_data.signal_table); interval=10^-2)|>
        Partition(T,step)|>
        Enumerate()|>
        Map() do (part_idx,part)
            sequence = DataFrame(part)
            transform!(sequence,eachindex=>:i)
            GBPAlgorithm.reset_msg!(decode_iter!)
            DataPrep.memory_factor!(factors.M,task_data.model_table,sequence.Rx)
            collapse_prior!(factors,sequence,T,step,part_idx)
            decode_iter!(factors)|>Drop(niter-1)|>Take(1)|>collect|>only
            GBPAlgorithm.beliefs!(factors,decode_iter!)
            transform!(sequence,:Ts=>(x->copy(factors.Rs))=>:Rs)
            transform!(sequence,[:Ts,:Rs]=>((t,r)->t.!=r)=>:error)
            progress_info_msg(prog,showprogressinfo)
            get_results(sequence,step,part_idx)
        end|>
        Take(n)|>
        foldxl(vcat)
    return select!(R,:Tx,:Rx,:is_pilot,:Ts,:Rs,:error)
end

function saveresults(data,task_info)
    (;power,with_noise,pilots,pilots_period,is_simulation,number_of_mixture_components) = task_info
    (;step,T) = task_info.alg_params
    (;savedir) = task_info.file_params
    prefix = is_simulation ? "simulation_decoded_points" : "decoded_points"
    x = (;power,with_noise,pilots,pilots_period,step,k=number_of_mixture_components,T)
    mkpath(datadir("results",savedir))
    fn = datadir("results",savedir,savename(prefix,x,"parquet"))
    d = select(data,
            :Tx=>ByRow(((x,y),)->(;Tx_x=x,Tx_y=y))=>AsTable,
            :Rx=>ByRow(((x,y),)->(;Rx_x=x,Rx_y=y))=>AsTable,

        )
    savedata(d,fn)
    save_info_msg(datadir("results",savedir))
    return data
end

# function summarize_results(res)
#     mapreduce(vcat,res) do r
#         (;task_info,results) = r
#         (;T,step) = task_info.alg_params
#         (;power,with_noise,pilots,pilots_period,number_of_mixture_components) = task_info
#         ber_gbp=(results.error./4)|>mean
#         (;power,with_noise,pilots,pilots_period,step,number_of_mixture_components,sequence_length=T,ber_gbp)
#     end|>DataFrame
# end

# fix_variance(task_info) = (;task_info..., noise_var=(task_info.noise_sigma*task_info.noise_scale*2)^2)

function run_it(; 
        power_vals=-2:8,with_noise_vals=false:true,pilots_vals=1:1,k_vals=2:2,
        T,w,niter,pilots_period,step,is_simulation=false,
        load_take=1_000_000,load_skip=0,should_save=true,savedir="",
        showprogressinfo=true,
        start_msg="Running GBP decoder for all powers"
    )
    decoding_tasks = decoding_tasks_iter(
        power_vals,with_noise_vals,pilots_vals,k_vals;
        pilots_period,
        is_simulation,
        alg_params=(;T,w,niter,step,showprogressinfo),
        file_params=(;load_take,load_skip,should_save,savedir,date=today())
    )

    @info "$(start_msg)"
    t0 = now()
    res = mapreduce(vcat,decoding_tasks) do task_info
        (;load_take,load_skip,should_save) = task_info.file_params
        task_info.alg_params.showprogressinfo && task_info_msg(task_info)
        @chain task_info begin
            load_problem_data(_,load_take,load_skip)
            add_pilots_info
            solve_decoding_task
            should_save ? saveresults(_,task_info) : _
            (;task_info,results=_)
        end
    end
    tf = now()
    @info "Done." "Elapsed time"=canonicalize(tf-t0)

    # d = summarize_results(res)
   
    # if should_save
    #     dir = datadir("results",savedir)
    #     @info "Saving summary..." "Directory"=dir
    #     mkpath(dir)
    #     @assert ispath(dir)
    #     x = (;T,step,niter,pilots_period,load_take,load_skip)
    #     prefix = is_simulation ? "simulation_summary" : "summary"
    #     fn = joinpath(dir,savename(prefix,x,"parquet";equals="_",connector="-"))
    #     d|>savedata(fn)
    # end

    println("\n")
    return (;#=summary=d,=#results=res)
end