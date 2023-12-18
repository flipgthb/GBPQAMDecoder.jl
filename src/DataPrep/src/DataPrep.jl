module DataPrep

using Chain, DataFrames, Distributions, LinearAlgebra, NamedTupleTools, Transducers

function parse_model_data(model_data)
	mat22(x...) = reshape(vcat(x...),(2,2))
	
	points = map([:left,:central,:right]) do p
		AsTable(Regex("$p"))=>ByRow(Map(Int)⨟foldxl(tuple))=>Symbol(p,:_point)
	end

	params = mapreduce(vcat,1:5) do k
		mapreduce(vcat,1:k) do m
			[Regex("h$(k)_mean\\d_$(m)")=>ByRow(vcat)=>Symbol(:h,k,:_mean_,m),
			 Regex("h$(k)_cov\\d+_$(m)")=>ByRow(mat22)=>Symbol(:h,k,:_cov_,m),
			 Regex("h$(k)_prob_$(m)")=>ByRow(identity)=>Symbol(:h,k,:_prob_,m)]
		end
	end

	group_params = mapreduce(vcat,[:mean,:cov,:prob]) do v
		map(1:5) do k
			Regex("h$(k)_$(v)")=>ByRow(tuple⨟collect)=>Symbol(:h,k,:_,v,:s)
		end
	end
	
	group_mixtures = mapreduce(vcat,1:5) do k
		Regex("h$(k)")=>ByRow(namedtuple(:means,:covs,:probs))=>Symbol(:mix,k)
	end

	@chain model_data begin
		select!(points...,params...)
		select!(r"point"=>ByRow(namedtuple(:l,:c,:r))=>:triplet,group_params...)
		select!(:triplet,group_mixtures...)
	end	
end

@info "Sigma is variance mode"
function memory_and_noise_model_table(model_table,k,σ)

	model(μ,σ,w) = MixtureModel(MvNormal,collect(zip(μ,σ)),w)
	
	df = DataFrames.select(model_table,:triplet,Regex("mix$(k)"))
	@chain df begin
		select!(:triplet,r"mix"=>ByRow(identity)=>AsTable)
		transform!(
			[:means,:triplet]=>ByRow((m,t)->map(x->x+collect(t.c),m))=>:means,
			:covs=>ByRow(c->map(x->x+(σ)*I,c))=>:covs,
			:probs=>ByRow(x->normalize!(x,1))=>:probs
		)
		select!(
			:triplet,
			[:means,:covs,:probs]=>ByRow(model)=>:model
		)
	end
end

function memory_factor_table(mt,Rx)
	@chain mt begin
		DataFrames.select(
			:triplet,
			:model=>ByRow() do m
				map(xt->pdf(m,xt),Rx)
			end=>AsTable,
			Not(:model)
		)
		DataFrames.stack(r"x"; variable_name=:time_axis, value_name=:M)
		select!(
			[:triplet,:time_axis]=>ByRow() do lcr,tx
				t = parse(Int,match(r"x(?<t>\d+)",tx)["t"])
				(;l,c,r) = lcr
				CartesianIndex(l,c,r,t)
			end=>:index,
			:M
		)
	end
end

function triplet_constellation_to_symbol(mt,qam_encoding)
	@chain mt begin
		rename!(:triplet=>:triplet_positions)
		transform!(
			:triplet_positions=>ByRow(x->(;l=qam_encoding[x.l],c=qam_encoding[x.c],r=qam_encoding[x.r]))=>:triplet
		)
	end
end

function astensor!(M,mt; N=16)
	foreach(row->M[row.index] = row.M,eachrow(mt))
	return M
end

function memory_factor!(M,mt,q,k,Rx,σ)
	@chain mt begin
		memory_and_noise_model_table(k,σ)
		triplet_constellation_to_symbol(q)
		memory_factor_table(Rx)
		astensor!(M,_)
	end
end

function memory_factor(mt,q,k,Rx,σ)
	N = length(q)
	T = length(Rx)
	M = ones(N,N,N,T)
	memory_factor!(M,mt,q,k,Rx,σ)
end

function parse_signal_data(st)
	select!(st,
		AsTable(r"orig")=>ByRow(Map(x->round(Int,x))⨟foldxl(tuple))=>:Tx,
		r"shifted"=>ByRow(vcat)=>:Rx
	)
end

function signal_constellation_to_symbol(st,qam_encoding)
	transform!(st,
		:Tx=>ByRow(x->qam_encoding[x])=>:Ts
	)
end

function get_qam_encoding(constellation_data)
	@chain constellation_data begin
		select!(
			AsTable(r"orig")=>ByRow(Map(Int)⨟foldxl(tuple))=>:point,
			:symbol
		)
		combine(
			[:point,:symbol]=>ByRow(=>)=>:qam_encoding
		)
		Dict(_.qam_encoding)
	end
end

function get_noise_info(noise_data,power,with_noise)
	@chain noise_data begin
		subset(:pdbm=>ByRow(p->p==power))
		eachrow(_)|>
			Map() do r
				power=r.pdbm
				sigma = with_noise ? r.sigma : 0.0
				(;power,sigma,scale=r.scale,with_noise)
			end
		only
	end
end

end # module DataPrep
