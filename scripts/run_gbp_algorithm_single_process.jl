using DrWatson
@quickactivate "GBPQAMDecoder"

using Chain,
    DataFrames,
    Dates,
    LinearAlgebra,
    Parquet2,
    ProgressMeter,
    Random,
    StatsBase,
    Transducers
using DataPrep
using GBPAlgorithm

include(srcdir("data_utilities.jl"))
include(srcdir("decoding.jl"))

warm_up_results = let
    run_it(;
        power_vals=0:1,with_noise_vals=true:true,pilots_vals=1:1,k_vals=2:2,
        T=10,w=0.25,niter=10,pilots_period=100,step=8,is_simulation=false,
        load_take=200,load_skip=0,should_save=false,savedir="test_$(today())",
        start_msg="Warmming up...", showprogressinfo=false
    );

    run_it(;
        power_vals=0:1,with_noise_vals=true:true,pilots_vals=1:1,k_vals=2:2,
        T=10,w=0.25,niter=10,pilots_period=100,step=8,is_simulation=false,
        load_take=200,load_skip=0,should_save=true,savedir="test_$(today())",
        start_msg="one more, this time saving...", showprogressinfo=false
    );
end

println("")

print_help() = println("""
Run `run_it(; 
        power_vals=-2:8,with_noise_vals=false:true,pilots_vals=1:1,k_vals=2:2,
        T,w,niter,pilots_period,step,is_simulation=false,
        load_take=1_000_000,load_skip=0,should_save=true,savedir="",
        showprogressinfo=true,
        start_msg="Running GBP decoder for all powers"
    )`
 to run the GBP decoding algorithm for all available power levels and noise conditions given:
    - each windowed sequence with length `T` and overlap `step` in the beggining 
    - using a mixture with `k` Gaussian components                               (k in {1,2,3,4,5})
    - using a gradient descent step of `w`                                       (w in 0..1) 
    - on signal generated with `pilots` waves every `pilots_perior` symbols      (pilots in {0,1,2,3})
    - and optionally, if `should_save` saves it to a subdirectory `datadir("results",savedir)`
    - `load_take` and `load_skip` refer to how many symbols to load from the signal file or generate
        for a simulation
    - `is_simulation` to load the model information but not the signal file and generate a signal with
        the specified parameters using the respective memory and noise model
    - `start_msg` can be used to identify the current task in the system log files
    - `showprogressinfo` can be used to toggle the display of parameters and progress

Thw warm up results are assigned to `warm_up_results`, with a summary for all the datasets and the
decoding results split per dataset.

Run `print_help()` to show this message again. 
""")

print_help()