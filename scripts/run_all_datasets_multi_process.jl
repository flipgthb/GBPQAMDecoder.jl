using Distributed
if nprocs() == 1 
	addprocs()
end

@everywhere using DrWatson
@everywhere begin
	@quickactivate "GBPQAMDecoder"
end

@everywhere import DataPrep
@everywhere import GBPAlgorithm

@everywhere using Chain, DataFrames, NamedTupleTools, Parquet2, StatsBase, Transducers

@everywhere include(srcdir("utils.jl"))

@everywhere include(srcdir("pipeline_functions.jl"))



function run_all_datasets_multi_process(number_of_sequences=2, k=2, sequence_length=2; niter=2, decfunc=GBPAlgorithm.decode)
	all_dataset_params(sequence_length,number_of_sequences,k)|>
		MapCat() do params
			setup_decoding_tasks(; params...)|>
				foldxl(vcat)	
		end|>
		Map(decoding_results(niter,decfunc))|>
		dcollect|>
		DataFrame|>
		tidy_results|>
        savedata(datadir("exp_pro","nseq_$(number_of_sequences)_k_$(k)_T_$(sequence_length)_results.parquet"))
end

@info "Running a smaller batch to compile"
run_all_datasets_multi_process();
@info "Done. Run `run_all_datasets_multi_process(nseq,k,T;niter,decfunc)` with the desired parameters." 