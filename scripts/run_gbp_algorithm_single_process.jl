using DrWatson
@quickactivate :GBPQAMDecoder

using Chain,
    DataFrames,
    Dates,
    JLD2,
    LinearAlgebra,
    Parquet2,
    ProgressMeter,
    Random,
    StatsBase,
    Transducers

warm_up_results = let
    test_tasks = decoding_tasks_list(;
        power=[-1,1],with_noise=true,pilots=0,k=2,
        load_batches=2
    )

    results = map(test_tasks) do task_info
        produce_or_load(solve_decoding_task,task_info,datadir("results","testing_script"); force=true)
    end
end;

println("")