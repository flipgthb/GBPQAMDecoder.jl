using DrWatson
@quickactivate "GBPQAMDecoder"

import DataPrep
import GBPAlgorithm

using Chain, DataFrames, Dates, NamedTupleTools, Parquet2, StatsBase, Transducers

include(srcdir("utils.jl"))
include(srcdir("pipeline_functions.jl"))

datadilename(n,k,T;prefix="") = "$(prefix)_nseq_$(n)_k_$(k)_T_$(T)_results.parquet"

function run_all_datasets_single_process(number_of_sequences=3, k=2, sequence_length=5; niter=10, decfunc=GBPAlgorithm.decode, dir="",prefix="")
	t0 = now()
    mkpath(datadir("exp_pro",dir))
    @info "Starting $(t0)"
    r = all_dataset_params(sequence_length,number_of_sequences,k)|>
		MapCat() do params
			setup_decoding_tasks(; params...)|>
				foldxl(vcat)	
		end|>
		Map(decoding_results(niter,decfunc))|>
		DataFrame|>
		tidy_results|>
        savedata(datadir("exp_pro",dir,datadilename(number_of_sequences,k,sequence_length;prefix)))
    tf = now()
    @info "Finished $(tf). Took $(tf - t0)"
    return r
end

@info "Running a smaller batch to compile"
run_all_datasets_single_process(;prefix="test");
@info "... one more..."
run_all_datasets_single_process(;prefix="test");
@info "Done. Run `run_all_datasets_single_process(nseq,k,T;niter,decfunc,dir,prefix)` with the desired parameters." 