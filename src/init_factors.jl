uniform_prior!(P,M) = fill!(P,1.0)|>GBPAlgorithm.msg_closure!
uniform_prior(M) = (N=size(M,1); T=size(M,4); P=Array{Float64,2}(undef,N,T); uniform_prior!(P,M))

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

function (f::InitFactors)(st,(mt,q,k,σ)::NTuple{4})
	df = DataFrame(st)
	(;M,inverse) = DataPrep.memory_factor(mt,q,k,df.Rx,σ)|>f.modify_M
	P = f.init_P(M)
	return (;M,P,inverse)
end

(f::InitFactors)(mt,q,k,σ) = Base.Fix2(f,(mt,q,k,σ))

function (f::InitFactors)(st,(M,P,mt,q,k,σ)::NTuple{6})
	df = DataFrame(st)
	(;M,inverse) = DataPrep.memory_factor!(M,mt,q,k,df.Rx,σ)|>f.modify_M
	P = f.init_P(M)
	return (;M,P,inverse)
end

(f::InitFactors)(M,P,mt,q,k,σ) = Base.Fix2(f,((M,P),mt,q,k,σ))

function memory_factor_dummy_sandwich(M,(np,na)::Tuple{Int,Int})
	d = size.((M,),GBP.msg_axes(M))
	T = size(M,ndims(M))
	Ma = ones((d...,na))
	Mp = ones((d...,np))
	inverse(x) = getindex(x,(Colon() for _ in 1:(ndims(x)-1))...,(1+np):(T+np))
	return (;M=cat(Mp,M,Ma; dims=ndims(M)),inverse)
end

memory_factor_dummy_sandwich(np::Int,na::Int) = Base.Fix2(memory_factor_dummy_sandwich,(np,na))