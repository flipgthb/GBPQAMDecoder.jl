module GBPQAMDecoder

using DrWatson
@quickactivate "GBPQAMDecoder"

export solve_decoding_task, load_problem_data, decoding_tasks_list, DecodingTask, AlgorithmParams

using Chain,
    DataFrames,
    Dates,
    LinearAlgebra,
    Parquet2,
    ProgressMeter,
    Random,
    StatsBase,
    Transducers

using DataPrep     # local module
using GBPAlgorithm # local module

include(srcdir("decoding_tasks.jl"))
include(srcdir("signal_simulator.jl"))
include(srcdir("decoding.jl"))

end # end Decoding module