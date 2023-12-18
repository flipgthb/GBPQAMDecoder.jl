function setup_decoding_tasks(; 
    power, with_noise, with_pilot_wave, k, sequence_length, w, 
    number_of_sequences, step=sequence_length, 
    init_factors=default_init_factors(),extra...
)
    (;constellation_data,noise_data,model_data,signal_data) = problem_data(power, with_noise, with_pilot_wave)

    noise_info=DataPrep.get_noise_info(noise_data,power,with_noise)

    qam_encoding=DataPrep.get_qam_encoding(constellation_data)

    model_table=model_data|>DataPrep.parse_model_data

    signal_table = @chain signal_data begin
        DataPrep.parse_signal_data
        DataPrep.signal_constellation_to_symbol(qam_encoding)
    end

    signal_table|>
        Tables.rowtable|>
        Partition(sequence_length,step)|>
        Take(number_of_sequences)|>
        Zip(Map(identity),
            Map(init_factors(model_table,qam_encoding,k,noise_info.sigma))
        )|>
        Map() do (st,factors)
            signal_info = let
                df = DataFrame(st)
                (;Ts=df.Ts,Tx=df.Tx,Rx=df.Rx)
            end
            (;noise_info,signal_info,task=(;factors...,w),extra=(;k,with_pilot_wave,extra...))
        end
end

function decoding_results(task_info,(niter,decoding_function)::Tuple{Int,<:Function})
    (;M,P,w,inverse) = task_info.task
    (;Rs) = decoding_function(M,P,w)|>
            Drop(niter-1)|>
            Take(1)|>
            collect|>only
    (;noise_info,extra) = task_info
    (;Ts) = task_info.signal_info
    (;w) = task_info.task
    (;noise_info...,extra...,w,Ts,Rs=inverse(Rs))
end

# function decoding_results(params,(niter,decoding_function)::Tuple{Int,<:Function})
# 	setup_decoding_tasks(;params...)|>
# 		Map() do task_info
# 			(;M,P,w,inverse) = task_info.task
# 			(;Rs) = decoding_function(M,P,w)|>
# 					Drop(niter-1)|>
# 					Take(1)|>
# 					collect|>only
# 			(;noise_info,extra) = task_info
# 			(;Ts) = task_info.signal_info
# 			(;w) = task_info.task
# 			(;noise_info...,extra...,w,Ts,Rs=inverse(Rs))
# 		end
# end

decoding_results(niter::Int,decoding_function::F) where {F<:Function} = Base.Fix2(decoding_results,(niter,decoding_function))

function tidy_results(res; bitpersymbol=4)
	@chain res begin
		transform!(
			[:Ts,:Rs]=>ByRow((x,y)->count(x.!=y))=>:error_count,
			:Ts=>ByRow(length)=>:signal_length,
		)
		transform!(
			[:error_count,:signal_length]=>ByRow((e,T)->e/(bitpersymbol*T))=>:ber,
			:Ts=>ByRow((x->x.-1)⨟vec2nt(:transmited_t))=>AsTable,
			:Rs=>ByRow((x->x.-1)⨟vec2nt(:decoded_t))=>AsTable
		)
		select!(Not([:Ts,:Rs]))
	end
end

function results_stats(res)
	stats = combine(res,
		:power=>unique⨟only,
		:with_noise=>unique⨟only,
		:with_pilot_wave=>unique⨟only,
		:k=>unique⨟only,
		:ber=>mean
		; renamecols=false
	)

	rename!(stats,
		:power=>:power_dbm,
		:k=>:number_of_gaussians
	)

	return stats
end

function run_all_datasets(sequence_length,number_of_sequences,k; decoding_function, niter, collector=collect, kw...)
	all_dataset_params(sequence_length,number_of_sequences,k, kw...)|>
		Map(decoding_results(niter,decoding_function)⨟collector⨟DataFrame⨟tidy_results)
end