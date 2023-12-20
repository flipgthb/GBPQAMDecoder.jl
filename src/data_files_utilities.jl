
savedata(data,fn) = (Parquet2.writefile(fn,data); data)
savedata(fn::String) = Base.Fix2(savedata,fn)

model_data_file(power) = datadir(
	"exp_raw","model_data",
	"model_parameters_pdbm_$(power)_wo_noise.parquet"
)

signal_data_file(power,with_noise::Bool,with_pilot_wave::Bool) = datadir(
	"exp_raw","signal_data",
	"points_pdbm_$(power)_$(with_noise ? "w" : "wo")_noise_$(with_pilot_wave ? "w" : "wo")_pilotwave.parquet"
)

noise_data_file() = datadir("exp_raw","noise_info.parquet")

constellation_data_file() = datadir("exp_raw","constellation_info.parquet")

dataset(fn) = Parquet2.Dataset(fn)|>DataFrame

function decoding_task_info(;
		power::Int,
		with_noise::Bool,
		with_pilotwave::Bool=true,
		sequence_length::Int,
		number_of_sequences::Int,
		number_of_mixture_components::Int,
		sequence_step::Int=sequence_length,
		extra...
	)

	model_file = model_data_file(power)
	signal_file = signal_data_file(power,with_noise,with_pilotwave)
	encoding_file = constellation_data_file()

	(;sigma,scale) = @chain noise_data_file() begin
		dataset
		subset(:pdbm=>ByRow(p->p==power))
		map(eachrow(_)) do r
				power=r.pdbm
				sigma = with_noise ? r.sigma : 0.0
				(;power,sigma,scale=r.scale,with_noise)
			end
		only
	end

	task_info = (;
		power,
		with_noise,
		with_pilotwave,
		noise_sigma=sigma, 
		noise_var=sigma^2,
		noise_scale=scale,
		number_of_mixture_components,
		sequence_length,
		number_of_sequences,
		sequence_step,
		model_file,
		signal_file,
		encoding_file,
		extra...
	)

	return task_info
end

# function load_problem_data(task_info)
# 	qam_encoding = dataset(task_info.encoding_file)|>
# 					DataPrep.get_qam_encoding

# 	k = task_info.number_of_mixture_components
# 	σ² = task_info.noise_var

# 	model_table = @chain task_info.model_file begin
# 		dataset
# 		DataPrep.parse_model_data(k,σ²,qam_encoding)
# 	end

# 	signal_table = @chain task_info.signal_file begin
# 		dataset
# 		DataPrep.parse_signal_data(qam_encoding)
# 	end

# 	return (;task_info, qam_encoding, model_table, signal_table)
# end

function load_problem_data(task_info)
	qam_encoding = dataset(task_info.encoding_file)|>
					DataPrep.get_qam_encoding

	# k = task_info.number_of_mixture_components
	# σ² = task_info.noise_var

	model_table = @chain task_info.model_file begin
		dataset
		DataPrep.parse_model_data#(k,σ²,qam_encoding)
	end

	signal_table = @chain task_info.signal_file begin
		dataset
		DataPrep.parse_signal_data#(qam_encoding)
		DataPrep.signal_constellation_to_symbol(qam_encoding)
	end

	return (;task_info, qam_encoding, model_table, signal_table)
end