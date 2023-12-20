using DrWatson
@quickactivate "GBPQAMDecoder"

using Chain, DataFrames, Parquet2
include(srcdir("data_files_utilities.jl"))

function generate_task_info_table(powers;with_pilotwave,T,nseq,k,step=T)
	map(((p,n) for p in powers for n in false:true)) do (power,with_noise)
		decoding_task_info(;
			power,with_noise,with_pilotwave,
			sequence_length=T,number_of_sequences=nseq,
			number_of_mixture_components=k,
			sequence_step=step
		)
	end|>DataFrame
end

power_levels = -2:8
wwo(x) = x ? "w" : "wo" 

_fname_(k,T,nseq,step,with_pilotwave) = join(
        ["decoding_tasks_k_$(k)","T_$(T)","nseq_$(nseq)",
            "step_$(step)","$(wwo(with_pilotwave))_pilotwave",
            "parquet"],
        "_","."
    )

let T=5,nseq=3,step=5,with_pilotwave=true,k=2
    @info "Generating tasks for testing"
    generate_task_info_table(power_levels;with_pilotwave,T,nseq,k,step=T)|>
        savedata(datadir("exp_raw","test_"*_fname_(k,T,nseq,step,with_pilotwave)))
end

let Ts=(5,10,50), with_pilotwave=true, nseq=2500, k=2
    @info "Generating files for `T` in $(Ts) "
    foreach((5,10,50)) do T
        step=T
        generate_task_info_table(power_levels;with_pilotwave,T,nseq,k,step)|>
            savedata(datadir("exp_raw",_fname_(k,T,nseq,step,with_pilotwave)))
    end
end

@info "Decoding tasks info tables saved in $(datadir("exp_raw"))."