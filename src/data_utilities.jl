
savedata(data,fn) = (Parquet2.writefile(fn,data); data)
savedata(fn::String) = Base.Fix2(savedata,fn)

model_data_file(power) = datadir(
	"qam_data",
	"model_power_$(power)_wo_noise.parquet"
)

signal_data_file(power::Int,with_noise::Bool,pilots::Int) = datadir(
	"qam_data",
	"points_power_$(power)_pilots_$(pilots)_$(with_noise ? "w" : "wo")_noise.parquet"
)

noise_data_file() = datadir("noise_info.parquet")

constellation_data_file() = datadir("constellation_info.parquet")

dataset(fn) = Parquet2.Dataset(fn)|>DataFrame

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

function simulate_signal_table(task_info,model_table,qam_encoding)
	n = task_info.symbol_sequence_length
	(;pilots,pilots_period,pilot_point) = task_info
	Ts = rand_qam_symbol(qam_encoding,n)
	pilot_symbol = qam_encoding[pilot_point]
	place_pilot_symbols!(Ts,pilot_symbol,pilots_period,pilots)
	Tx = qam_encode(Ts,qam_encoding)
	Rx = simulate_transmission(Ts,model_table,qam_encoding)
	return DataFrame(:Ts=>Ts,:Tx=>Tx,:Rx=>Rx)
end

# function decoding_task_info(;
# 		power::Int,
# 		with_noise::Bool,
# 		pilots::Int,
# 		number_of_mixture_components::Int
# 	)

# 	model_file = model_data_file(power)
# 	signal_file = signal_data_file(power,with_noise,pilots)
# 	encoding_file = constellation_data_file()

# 	(;sigma,scale) = @chain noise_data_file() begin
# 		dataset
# 		subset(:pdbm=>ByRow(p->p==power))
# 		map(eachrow(_)) do r
# 				power=r.pdbm
# 				sigma = with_noise ? r.sigma*r.scale : 0.0
# 				(;power,sigma,scale=r.scale,with_noise)
# 			end
# 		only
# 	end

# 	task_info = (;
# 		power,
# 		with_noise,
# 		pilots,
# 		noise_sigma=sigma, 
# 		noise_var=sigma^2,
# 		noise_scale=scale,
# 		number_of_mixture_components,
# 		model_file,
# 		signal_file,
# 		encoding_file
# 	)

# 	return task_info
# end

# maxparts(nsyms,partlen,step) = floor(Int,1 + (nsyms - partlen)/step)
# maxparts(seqdf::DataFrame,partlen,step) = maxparts(size(seqdf,1),partlen,step)

function decoding_task_info(;
		power::Int,
		with_noise::Bool,
		pilots::Int,
		pilots_period::Int,
		pilot_point::Tuple{Int,Int}=(1,1),
		is_simulation::Bool=false,
		number_of_mixture_components::Int,
		extra...
	)

	model_file = model_data_file(power)
	encoding_file = constellation_data_file()
	signal_file = signal_data_file(power,with_noise,pilots)

	(;sigma,scale) = @chain noise_data_file() begin
		dataset
		subset(:pdbm=>ByRow(p->p==power))
		map(eachrow(_)) do r
				power=r.pdbm
				sigma = with_noise ? r.sigma*r.scale : 0.0
				(;power,sigma,scale=r.scale,with_noise)
			end
		only
	end

	task_info = (;
		power,
		with_noise,
		pilots,
		pilots_period,
		pilot_point,
		noise_sigma=sigma, 
		noise_var=sigma^2,
		noise_scale=scale,
		number_of_mixture_components,
		is_simulation,
		extra...,
		model_file,
		encoding_file,
		signal_file
	)

	return task_info
end

function decoding_tasks_iter(
		power_vals,with_noise_vals,pilots_vals,k_vals;
		pilots_period::Int,
		pilot_point::Tuple{Int,Int}=(1,1),
		is_simulation::Bool=false,
		extra...
	)

	return (
        decoding_task_info(;
			power,with_noise,pilots,number_of_mixture_components,
			pilots_period,pilot_point,is_simulation,extra...
		)
        for power in power_vals
        for with_noise in with_noise_vals
		for pilots in pilots_vals
		for number_of_mixture_components in k_vals
    )
end

function load_problem_data(task_info,load_take=1_000_000,load_skip=0)
	qam_encoding = dataset(task_info.encoding_file)|>
					DataPrep.get_qam_encoding

	k = task_info.number_of_mixture_components
	noise_var = task_info.noise_var

	model_table = @chain task_info.model_file begin
		dataset
		DataPrep.parse_model_data(k,noise_var,qam_encoding)
	end

	signal_table = if !task_info.is_simulation
			@chain task_info.signal_file begin
			# dataset
			Parquet2.Dataset
			Tables.rows
			_|>Drop(load_skip)|>Take(load_take)|>DataFrame
			DataPrep.parse_signal_data(qam_encoding)
		end
	else
		@warn "Simulating signal..."
		symbol_sequence_length=min(load_take,size(dataset(task_info.signal_file),1))
		simulate_signal_table((;task_info...,symbol_sequence_length),model_table,qam_encoding)
	end

	return (;task_info, qam_encoding, model_table, signal_table)
end