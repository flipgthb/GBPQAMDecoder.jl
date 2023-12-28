struct Bits{N}
    b::BitVector
end

Bits{N}(x::Int) where N = ((parse(Bool,b) for b in last(bitstring(x),N))...,)|>BitArray|>Bits{N}

Base.:⊻(x::Bits{N},y::Bits{N}) where N = Bits{N}(x.b .⊻ y.b)

asrate(x::Bits{N}) where N = x.b.//N

ber(x::Bits,y::Bits) = sum(asrate(x ⊻ y))
ber((x,y)::Tuple{Int,Int},N::Int) = ber(Bits{N}(x),Bits{N}(y))
ber(N::Int) = Base.Fix2(ber,N)∘tuple

ber_approx((x,y)::Tuple{Int,Int},N::Int) = (x!=y)//N
ber_approx(N::Int) = Base.Fix2(ber,N)∘tuple

function compute_ber!(results)
    select(results,
        [:Ts,:Rs]=>ByRow((x,y)->ber(4)(x-1,y-1))=>:ber_exact,
        [:Ts,:Rs]=>ByRow(ber_approx(4))=>:ber_approx
    )
end

function ber_statistics(data)
    rdf = combine(compute_ber!(data["results"]),
        :ber_exact=>mean,:ber_approx=>mean;
        renamecols=false
    )
    insertcols!(rdf,
        1, (k=>v for (k,v) in data["decoding_task"] 
            if k in (:power,:with_noise,:pilots,:pilots_period,:k,:load_batches,:skip_batches))...,
        (k=>v for (k,v) in data["decoding_task"][:alg_params]
             if k in (:T,:step,:w,:niter))...
    )    
    return rdf|>Tables.rowtable|>only|>pairs|>collect
end

function summarize_results!(
            filename,
            folder;
            special_list = [ber_statistics,],
            black_list = ["gitcommit","gitpatch","script","path","results","decoding_task"],
            kw...
        )
    return select!(collect_results!(filename,folder; black_list, special_list, kw...))#, Not(:path))
end

summarize_results!(folder;kw...) = summarize_results!(joinpath(dirname(folder), "summary_$(basename(folder)).jld2"),folder;kw...)

function summarize_results(
        folder;
        special_list = [ber_statistics,],
        black_list = ["gitcommit","gitpatch","script","path","results","decoding_task"],
        kw...
    )
    return select!(collect_results(folder; black_list, special_list, kw...), Not(:path))
end