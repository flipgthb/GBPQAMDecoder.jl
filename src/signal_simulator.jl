rand_qam_symbol(qam_encoding,dims...) = rand(qam_encoding|>values|>collect,dims...)

function simulate_transmission(symbol_seq,model_table,qam_encoding)
	x = [rand_qam_symbol(qam_encoding);
		 symbol_seq;
    	 rand_qam_symbol(qam_encoding)]
    Ts = axes(x,1)
    triplets = [x[mod.(t-1:t+1,(Ts,))] for t in Ts][2:end-1]
    let m=model_table
        map(triplets) do (l,c,r)
            rand(m[(m.l.==l) .&& (m.c.==c) .&& (m.r.==r),:model][1])
        end
    end
end

qam_encode(s::Int,qam_encoding) = filter(p->last(p)==s,qam_encoding)|>first|>first
qam_encode(qam_encoding) = Base.Fix2(qam_encode,qam_encoding)
qam_encode(symbol_seq::Vector{Int},qam_encoding) = map(qam_encode(qam_encoding),symbol_seq)

function place_pilot_symbols!(symbol_seq,pilot_symbol,pilots_period,pilots)
	for i in eachindex(symbol_seq)
		if mod(i,1:pilots_period) <= pilots
			symbol_seq[i] = pilot_symbol
		end
	end
	return symbol_seq
end

function simulate_signal_table(task_info,symbol_sequence_length,model_table,qam_encoding)
	n = symbol_sequence_length
	(;pilots,pilots_period,pilot_point) = task_info
	Ts = rand_qam_symbol(qam_encoding,n)
	pilot_symbol = qam_encoding[pilot_point]
	place_pilot_symbols!(Ts,pilot_symbol,pilots_period,pilots)
	Tx = qam_encode(Ts,qam_encoding)
	Rx = simulate_transmission(Ts,model_table,qam_encoding)
	return DataFrame(:Ts=>Ts,:Tx=>Tx,:Rx=>Rx)
end