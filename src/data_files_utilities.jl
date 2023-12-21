
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

function decoding_task_info(;
		power::Int,
		with_noise::Bool,
		pilots::Int,
		number_of_mixture_components::Int
	)

	model_file = model_data_file(power)
	signal_file = signal_data_file(power,with_noise,pilots)
	encoding_file = constellation_data_file()

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
		noise_sigma=sigma, 
		noise_var=sigma^2,
		noise_scale=scale,
		number_of_mixture_components,
		model_file,
		signal_file,
		encoding_file
	)

	return task_info
end

function load_problem_data(task_info)
	qam_encoding = dataset(task_info.encoding_file)|>
					DataPrep.get_qam_encoding

	k = task_info.number_of_mixture_components
	noise_var = task_info.noise_var

	model_table = @chain task_info.model_file begin
		dataset
		DataPrep.parse_model_data(k,noise_var,qam_encoding)
	end

	signal_table = @chain task_info.signal_file begin
		dataset
		DataPrep.parse_signal_data(qam_encoding)
	end

	return (;task_info, qam_encoding, model_table, signal_table)
end