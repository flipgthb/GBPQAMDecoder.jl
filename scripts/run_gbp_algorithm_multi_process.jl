using Distributed
addprocs(4)

@everywhere using DrWatson

@everywhere begin
    @quickactivate :GBPQAMDecoder

    using Chain,
        DataFrames,
        Dates,
        JLD2,
        LinearAlgebra,
        Observables,
        ObservablePmap,
        Parquet2,
        ProgressMeter,
        Random,
        StatsBase,
        TerminalLoggers,
        Transducers
end

function move_cursor_up_while_clearing_lines(io,numlinesup)
    for _ in 1:numlinesup
        print(io,"\r\u1b[K\u1b[A")
    end
end
move_cursor_up_while_clearing_lines(numlinesup) = Base.Fix1(move_cursor_up_while_clearing_lines,stderr)

function printover(io::IO,s::AbstractString, color::Symbol = :color_normal)
    print(io,"\r")
    printstyled(io,s; color=color)
    # print(io,s)
    print(io,"\u1b[K")     # clear the rest of the line
end

printover(s::AbstractString,color::Symbol=:color_normal) = printover(stderr,s,color)

let
    test_tasks = decoding_tasks_list(;
        power=[-1,1],with_noise=[false,true],pilots=0,k=2,
        load_batches=1#,alg_params=(;T=10,step=8,showprogressinfo=false)
    )

    @info "Warming up..."
    obs, task = ologpmap(test_tasks; schedule_now=true, logger_f=TerminalLogger) do task_info
        produce_or_load(
            solve_decoding_task,task_info,datadir("results","testing_script");
            force=true,verbose=true,tag=false
        )
        GC.gc()
    end

    sleep(45)

    obs_func = on(obs; update=false, weak=true) do val
        n = split(val,'\n')
        s = val*"\n"^(21-n)
        move_cursor_up_while_clearing_lines(21)
        printover(s)
        flush(stderr)
    end
end;

GC.gc()

let
    test_tasks = decoding_tasks_list(;
    power=[-1,1],with_noise=[false,true],pilots=0,k=2,
    load_batches=250#,alg_params=(;T=10,step=8,showprogressinfo=false)
    )

    @info "...One more..."
    obs, task = ologpmap(test_tasks; schedule_now=true, logger_f=TerminalLogger) do task_info
        produce_or_load(
            solve_decoding_task,task_info,datadir("results","testing_script");
            force=true,verbose=true,tag=false
        )
        GC.gc()
    end

    sleep(1)

    # obs[] = join([rpad("",80) for _ in 1:nprocs()], '\n')
    obs_func = on(obs; update=false, weak=true) do val
        n = split(val,'\n')
        s = val*"\n"^(21-n)
        move_cursor_up_while_clearing_lines(21)
        printover(s)
        flush(stderr)
    end
end;

let
    test_tasks = decoding_tasks_list(;
    power=collect(-2:8),with_noise=[false,true],pilots=collect(0:3),k=collect(1:3),
    load_batches=2600#,alg_params=(;T=10,step=8,showprogressinfo=false)
    )

    @info "Running GBP for all cases"
    obs, task = ologpmap(test_tasks; schedule_now=true, logger_f=TerminalLogger) do task_info
        produce_or_load(
            solve_decoding_task,task_info,datadir("results","gbp_all");
            force=true,verbose=true,tag=false
        )
        GC.gc()
    end

    sleep(1800)

    obs_func = on(obs; update=false, weak=true) do val
        n = split(val,'\n')
        s = val*"\n"^(21-n)
        move_cursor_up_while_clearing_lines(21)
        printover(s)
        flush(stderr)
    end
end;


# let
#     test_tasks = decoding_tasks_list(;
#         power=[-1,1],with_noise=[false,true],pilots=0,k=2,
#         load_batches=1,alg_params=(;T=10,step=8,showprogressinfo=false)
#     )

#     p = Progress(length(test_tasks); desc="Warming up...", dt=10^-3, showspeed=true)
#     channel = RemoteChannel(() -> Channel{Bool}(), 1)

#     @sync begin # start two tasks which will be synced in the very end
#         # the first task updates the progress bar
#         @async while take!(channel)
#             next!(p)
#         end

#         # the second task does the computation
#         @async begin
#             @distributed (+) for task_info in test_tasks
#                 produce_or_load(
#                     solve_decoding_task,task_info,datadir("results","testing_script");
#                     force=true,verbose=false,tag=false
#                 )
#                 GC.gc()
#                 put!(channel, true) # trigger a progress bar update
#                 1
#             end
#             put!(channel, false) # this tells the printing task to finish
#         end
#     end

#     # progress_pmap(test_tasks; progress) do task_info
#     #     GC.gc()
#     #     produce_or_load(
#     #         solve_decoding_task,task_info,datadir("results","testing_script");
#     #          force=true,verbose=false,tag=false
#     #     )
#     # end

#     GC.gc()
# end;

# GC.gc()

# let
#     test_tasks = decoding_tasks_list(;
#         power=[-1,1],with_noise=[false,true],pilots=0,k=2,
#         load_batches=25,alg_params=(;T=10,step=8,showprogressinfo=false)
#     )

#     p = Progress(length(test_tasks); desc="... one more...", dt=10^-3, showspeed=true)
#     channel = RemoteChannel(() -> Channel{Bool}(), 1)

#     @sync begin # start two tasks which will be synced in the very end
#         # the first task updates the progress bar
#         @async while take!(channel)
#             next!(p)
#         end

#         # the second task does the computation
#         @async begin
#             @distributed (+) for task_info in test_tasks
#                 produce_or_load(
#                     solve_decoding_task,task_info,datadir("results","testing_script");
#                     force=true,verbose=false,tag=false
#                 )
#                 GC.gc()
#                 put!(channel, true) # trigger a progress bar update
#                 1
#             end
#             put!(channel, false) # this tells the printing task to finish
#         end
#     end

#     # progress_pmap(test_tasks; progress) do task_info
#     #     GC.gc()
#     #     produce_or_load(
#     #         solve_decoding_task,task_info,datadir("results","testing_script");
#     #          force=true,verbose=false,tag=false
#     #     )
#     # end

#     GC.gc()
# end;

println("")