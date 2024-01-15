function add_pilots_info!(signal_table,task_info)
    (;pilots,pilots_period) = task_info
    @chain signal_table begin
        transform!(eachindex=>:t)
        transform!(:t=>ByRow(t->mod(t,1:pilots_period)<=pilots)=>:is_pilot)
    end
end

function collapse_prior!(factors,sequence,T,step,part_idx)
    overlap = part_idx > 1 ? pairs(last(factors.Rs,T-step)) : ()
    pilots = (r.i=>r.Ts for r in eachrow(sequence) if r.is_pilot)
    GBPAlgorithm.collapse_prior!(factors,overlap...,pilots...)
    return factors
end

function load_problem_data(task_info)
    (;k,qam_encoding) = task_info
    (;noise_var) = task_info.noise_info
	model_table = @chain task_info.model_file begin
		dataset
		DataPrep.parse_model_data(k,noise_var,qam_encoding)
	end

    (;load_batches,skip_batches,pilots_period) = task_info
	signal_table = if !task_info.is_simulation
			@chain task_info.signal_file begin
			Parquet2.Dataset
			Tables.rows
			_|>Drop(skip_batches*pilots_period)|>Take(load_batches*pilots_period)|>DataFrame
			DataPrep.parse_signal_data(qam_encoding)
            add_pilots_info!(task_info)
		end
	else
		@warn "Simulating signal..."
		symbol_sequence_length=min(load_batches*pilots_period,262_000)
		add_pilots_info!(
            simulate_signal_table(
                task_info,
                symbol_sequence_length,
                model_table,
                qam_encoding
            ),
            task_info
        )
	end

    return (;model_table,signal_table)
end

function progress_info_msg(prog,task_info)
    (;power,with_noise,pilots,k) = task_info
    (;T,step,w,niter) = task_info.alg_params
    showvalues = [(:power,power),(:with_noise,with_noise),(:pilots,pilots),
                  (:k,k),(:T,T),(:step,step),(:w,w),(:niter,niter)]
    task_info.alg_params.showprogressinfo && next!(prog; showvalues)
end

# function task_info_msg(task_info)
#     (;T,step) = task_info.alg_params
#     (;power,with_noise,pilots,k) = task_info
#     @info "Running algorithm:" "Avg. Power [dBm]"=power "With 4.5dB noise"=with_noise "Pilots"=pilots "Number of mixture components"=k "Sequence length"=T "Step"=step
# end

function task_info_msg(task_info)
    (;T,step) = task_info.alg_params
    (;power,with_noise,pilots,k) = task_info
    msg = "Deocding: $(power)dBm with$(with_noise ? " " : "out ")noise, $(pilots) pilots,k=$(k),T=$(T),step=$(step)"
    rpad(msg,60)
end

function save_info_msg(dir)
    @info "Saved decoded points" "Directory"=dir
end

function get_noise_info(noise_data,power,with_noise)
    @chain noise_data begin
        subset(:pdbm=>ByRow(p->p==power))
        map(eachrow(_)) do r
            noise_sigma = with_noise ? r.sigma*r.scale : 0.0
            noise_var = noise_sigma^2
            (;noise_sigma,noise_var,noise_scale=r.scale)
        end
        only
    end
end

getresults(sequence,step,part_idx) = part_idx > 1 ? last(sequence,step) : sequence

maxparts(nsyms,partlen,step) = floor(Int,1 + (nsyms - partlen)/step)
maxparts(seqdf::DataFrame,partlen,step) = maxparts(size(seqdf,1),partlen,step)

function solve_decoding_task(task_info)
    (;model_table,signal_table) = load_problem_data(task_info)
    (;T,w,niter,step,showprogressinfo) = task_info.alg_params
    # showprogressinfo && task_info_msg(task_info)
    N = length(task_info.qam_encoding)
    decode_iter! = GBPAlgorithm.GBPDecoder(N,T,w)
    factors = GBPAlgorithm.Factors(N,T)
    n = maxparts(size(signal_table,1),T,step)
    prog = Progress(n;
            desc=task_info_msg(task_info),
            barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
            barlen=16,
            showspeed=true
        )
    R = withprogress(eachrow(signal_table); interval=10^-2)|>
        Partition(T,step)|>
        Enumerate()|>
        Map() do (part_idx,part)
            sequence = DataFrame(part)
            transform!(sequence,eachindex=>:i)
            GBPAlgorithm.reset_msg!(decode_iter!)
            DataPrep.memory_factor!(factors.M,model_table,sequence.Rx)
            collapse_prior!(factors,sequence,T,step,part_idx)
            decode_iter!(factors)|>Drop(niter-1)|>Take(1)|>collect|>only
            GBPAlgorithm.beliefs!(factors,decode_iter!)
            transform!(sequence,:Ts=>(x->copy(factors.Rs))=>:Rs)
            transform!(sequence,[:Ts,:Rs]=>((t,r)->t.!=r)=>:error)
            # progress_info_msg(prog,task_info)
            showprogressinfo && ProgressMeter.next!(prog)
            getresults(sequence,step,part_idx)
        end|>
        Take(n)|>
        foldxl(vcat)
    results = select!(R,
                :Tx=>ByRow(((x,y),)->(;Tx_x=x,Tx_y=y))=>AsTable,
                :Rx=>ByRow(((x,y),)->(;Rx_x=x,Rx_y=y))=>AsTable,
                :Ts,:Rs,:error,:is_pilot
            )

    decoding_task = decodingtask2dict(task_info)
    return @strdict decoding_task results
end