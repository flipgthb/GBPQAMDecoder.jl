module DataPrep

using Chain, DataFrames, Distributions, LinearAlgebra, NamedTupleTools, Transducers

noise_model(agwn_var) = c->c + agwn_var*I
noise_model(agwn_var,cs) = map(noise_model(agwn_var),cs)

function parse_model_data(model_data_table,k,noise_var,qam_encoding)
	@chain model_data_table begin
		select!(
			[r"left",r"central",r"right"].=>
				ByRow((x,y)->round.(Int,(x,y))).=>[:l_point,:c_point,:r_point],
			[Regex("h$(k)_mean\\d_$(m)") for m in 1:k].=>
				ByRow(vcat).=>[Symbol(:mean_,m) for m in 1:k],
			[Regex("h$(k)_cov\\d+_$(m)") for m in 1:k].=>
				ByRow() do x...
					reshape(vcat(x...),(2,2))
				end.=>[Symbol(:cov_,m) for m in 1:k],
			[Regex("h$(k)_prob_$(m)") for m in 1:k].=>
				ByRow(identity).=>[Symbol(:prob_,m) for m in 1:k]
		)
		select!(
			r"point",
			[r"mean",r"cov",r"prob"].=>ByRow(collect∘tuple).=>[:means,:covs,:probs],
		)
		select!(
			r"point",
			[:means,:covs,:probs,:c_point]=>
				ByRow() do means,covs,weights,p
					x_c = collect(p)
					agwm_cov = noise_var*I
					components = [MvNormal(μ + x_c,Σ + agwm_cov) for (μ,Σ) in zip(means,covs)]
					MixtureModel(components,weights)
				end=>:model
		)
		select!(
			r"point",
			[r"^l",r"^c",r"^r"].=>ByRow(x->qam_encoding[x]).=>[:l,:c,:r],
			:model,
		)
	end
end

function M_factor_table(mt,Rx)
	@chain mt begin
		DataFrames.select(
			:l,:c,:r,
			:model=>ByRow() do m
				map(xt->pdf(m,xt),Rx)
			end=>AsTable,
			Not(:model)
		)
		DataFrames.stack(r"x"; variable_name=:time_axis, value_name=:M)
		select!(
			[:l,:c,:r,:time_axis]=>ByRow() do l,c,r,tx
				t = parse(Int,match(r"x(?<t>\d+)",tx)["t"])
				CartesianIndex(l,c,r,t)
			end=>:index,
			:M
		)
	end
end

function astensor!(M,mft)
	foreach(row->M[row.index] = row.M,eachrow(mft))
	return M
end

function memory_factor!(M,mt,Rx)
	@chain mt begin
		M_factor_table(Rx)
		astensor!(M,_)
	end
end

function memory_factor(mt,Rx)
	N = round(Int,size(mt,1)^(1/3))#length(q)
	T = length(Rx)
	M = ones(N,N,N,T)
	memory_factor!(M,mt,Rx)
end

function parse_signal_data(st,qam_encoding)
	@chain st begin
		select!(
			AsTable(r"orig")=>ByRow(Map(x->round(Int,x))⨟foldxl(tuple))=>:Tx,
			r"shifted"=>ByRow(vcat)=>:Rx
		)
		transform!(
			:Tx=>ByRow(x->qam_encoding[x])=>:Ts
		)
	end
end

function get_qam_encoding(constellation_data)
	@chain constellation_data begin
		select!(
			r"orig"=>ByRow((x,y)->round.(Int,(x,y)))=>:point,
			:symbol
		)
		combine(
			[:point,:symbol]=>ByRow(=>)=>:qam_encoding
		)
		Dict(_.qam_encoding)
	end
end

# function get_noise_info(noise_data,power,with_noise)
# 	@chain noise_data begin
# 		subset(:pdbm=>ByRow(p->p==power))
# 		eachrow(_)|>
# 			Map() do r
# 				power=r.pdbm
# 				sigma = with_noise ? r.sigma : 0.0
# 				(;power,sigma,scale=r.scale,with_noise)
# 			end
# 		only
# 	end
# end

function parse_model_data_old_adapted(model_data,k,noise_var,qam_encoding)
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
		memory_and_noise_model_table_old_adapted(k,noise_var,qam_encoding)
	end	
end

function triplet_constellation_to_symbol_old_adapted(mt,qam_encoding)
	@chain mt begin
		rename!(:triplet=>:triplet_positions)
		transform!(
			:triplet_positions=>ByRow(x->(;l=qam_encoding[x.l],c=qam_encoding[x.c],r=qam_encoding[x.r]))=>:triplet
		)
	end
end

function memory_and_noise_model_table_old_adapted(model_table,k,σ²,qam_encoding)

	model(μ,σ,w) = MixtureModel(MvNormal,collect(zip(μ,σ)),w)
	
	df = DataFrames.select(model_table,:triplet,Regex("mix$(k)"))
	@chain df begin
		select!(:triplet,r"mix"=>ByRow(identity)=>AsTable)
		transform!(
			[:means,:triplet]=>ByRow((m,t)->map(x->x+collect(t.c),m))=>:means,
			:covs=>ByRow(c->map(x->x+σ²*I,c))=>:covs,
			:probs=>ByRow(x->normalize!(x,1))=>:probs
		)
		select!(
			:triplet,
			[:means,:covs,:probs]=>ByRow(model)=>:model
		)
		triplet_constellation_to_symbol_old_adapted(qam_encoding)

	end
end

end # module DataPrep