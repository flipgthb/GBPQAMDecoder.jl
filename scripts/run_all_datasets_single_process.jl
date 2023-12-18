using DrWatson
@quickactivate "GBPQAMDecoder"

import DataPrep
import GBPAlgorithm

using Chain, DataFrames, Dates, NamedTupleTools, Parquet2, StatsBase, Transducers

include(srcdir("utils.jl"))
include(srcdir("pipeline_functions.jl"))

function run_all_datasets_single_process(number_of_sequences=3, k=2, sequence_length=5; niter=10, decfunc=GBPAlgorithm.decode)
	t0 = now()
    @info "Starting $(t0)"
    r = all_dataset_params(sequence_length,number_of_sequences,k)|>
		MapCat() do params
			setup_decoding_tasks(; params...)|>
				foldxl(vcat)	
		end|>
		Map(decoding_results(niter,decfunc))|>
		# collect|>
		DataFrame|>
		tidy_results|>
        savedata(datadir("exp_pro","nseq_$(number_of_sequences)_k_$(k)_T_$(sequence_length)_results.parquet"))
    tf = now()
    @info "Finished $(tf). Took $(tf - t0)"
end

@info "Running a smaller batch to compile"
run_all_datasets_single_process();
@info "... one more..."
run_all_datasets_single_process();
@info "Done. Run `run_all_datasets_single_process(nseq,k,T;niter,decfunc)` with the desired parameters." 