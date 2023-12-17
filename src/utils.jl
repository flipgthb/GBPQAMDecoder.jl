
# datadir(p...) = joinpath("..","data",p...)

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

problem_data(power,with_noise,with_pilot_wave) = (;
	constellation_data=constellation_data_file()|>dataset,
	noise_data=noise_data_file()|>dataset,
	model_data=model_data_file(power)|>dataset,
	signal_data=signal_data_file(power,with_noise,with_pilot_wave)|>dataset
)

# ----
uniform_prior(M) = (N=size(M,1); T=size(M,4); ones(N,T)|>GBPAlgorithm.msg_closure!)

# function distance_matrix(Rx, Q)
# 	y = hcat(Rx...)
# 	x = sort!([v=>k for (k,v) in Q]; by=first)|>
# 		Map(last⨟collect)|>
# 		foldxl(hcat)
# 	return pairwise(Euclidean(),x,y)
# end

vec2nt(vec,tag) = namedtuple([Symbol(tag,i) for i in 1:length(vec)])(vec)
vec2nt(tag::Symbol) = Base.Fix2(vec2nt,tag)

reusable_parameters(; kw...) = (;
	nqam=16,w=0.5,with_pilot_wave=true,kw...
)

function all_dataset_params(sequence_length,number_of_sequences,k; kw...)
	y = (;sequence_length,number_of_sequences,k)
	((power=p,with_noise=n) for p in -2:8 for n in (false,true))|>
		Map(x->merge(x,y,reusable_parameters(; kw...)))	
end

struct InitFactors{FM,FP}
	modify_M::FM
	init_P::FP
end

identity_factor(M) = (;M,inverse=identity)

default_init_factors(fM=identity_factor,fP=uniform_prior) = InitFactors{typeof(fM),typeof(fP)}(fM,fP)

function (f::InitFactors)(st,(mt,q,k,σ))
	df = DataFrame(st)
	(;M,inverse) = DataPrep.memory_factor(mt,q,k,df.Rx,σ)|>f.modify_M
	P = f.init_P(M)
	return (;M,P,inverse)
end

(f::InitFactors)(mt,q,k,σ) = Base.Fix2(f,(mt,q,k,σ))

function memory_factor_dummy_sandwich(M,(np,na)::Tuple{Int,Int})
	d = size.((M,),GBP.msg_axes(M))
	T = size(M,ndims(M))
	Ma = ones((d...,na))
	Mp = ones((d...,np))
	inverse(x) = getindex(x,(Colon() for _ in 1:(ndims(x)-1))...,(1+np):(T+np))
	return (;M=cat(Mp,M,Ma; dims=ndims(M)),inverse)
end

memory_factor_dummy_sandwich(np::Int,na::Int) = Base.Fix2(memory_factor_dummy_sandwich,(np,na))

savedata(data,fn) = data|>tuple⨟
				Zip(Map(Base.Fix1(Parquet2.writefile,fn)),Map(identity))⨟
				only⨟
				last

# function results_data_file(params; dir="", 
# 	use_params=(:power,:with_noise,:with_pilot_wave,:k,:sequence_length,:number_of_sequences)
# )
# 	fn = NamedTupleTools.select(params,use_params)|>
# 	pairs|>
# 	Map() do (f,l)
# 		string.([f,l])
# 	end|>
# 	foldxl(vcat)|>
# 	x->join(x,"_")|>
# 	x->"results_$(x)_sequences.parquet"
# 	return datadir("exp_pro",dir,fn)
# end