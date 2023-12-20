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
			[:means,:covs,:probs]=>
				ByRow() do means,covs,weights
					agwm_cov = noise_var*I
					components = [MvNormal(μ,Σ + agwm_cov) for (μ,Σ) in zip(means,covs)]
					MixtureModel(components,weights)
					# params = collect(zip(means,noise_model(noise_var,covs)))
					# MixtureModel(MvNormal,params,weights)
				end=>:model
		)
		select!(
			[r"^l",r"^c",r"^r"].=>ByRow(x->qam_encoding[x]).=>[:l,:c,:r],
			:model,r"point"
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

end # module DataPrep