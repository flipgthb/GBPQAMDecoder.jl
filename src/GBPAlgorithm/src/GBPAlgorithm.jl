module GBPAlgorithm

using FGenerators, LinearAlgebra, StatsBase, Transducers, Tullio

symbol_axes(x) = ndims(x)
eachsymbol(x) = eachslice(x;dims=symbol_axes(x))
msg_axes(x) = tuple(1:(ndims(x)-1)...)
msg_axes(x::Vector) = 1
eachmsg(x) = eachslice(x;dims=msg_axes(x))

closure!(x) = normalize!(x,1)

msg_closure!(x) = (map!(closure!,eachsymbol(x),eachsymbol(x)); x)

msg_sum(x) = sum(x; dims=msg_axes(x))

msg_init(dims...) = fill(1.0,dims...)|>msg_closure!
function msg_init(M::Array{<:Real,4})
	N, T = size(M,1),size(M,4)
	return (s=msg_init(N,T),d=msg_init(N,T),A=msg_init(N,N,T),S=msg_init(N,N,T))
end

uniform_prior!(P::Matrix) = fill!(P,1.0)|>msg_closure!
uniform_prior(M::Array{<:Real,4}) = (N=size(M,1); T=size(M,4); P=Array{Float64,2}(undef,N,T); uniform_prior!(P))

function gbp_equations!(msgs_up,msgs,M,P)
	(;s,d,A,S) = msgs

	@tullio msgs_up.s[c,t] = P[l,mod(t-1)]*s[l,mod(t-1)]*A[l,c,t]*S[l,c,t]
	@tullio msgs_up.d[c,t] = P[r,mod(t+1)]*d[r,mod(t+1)]*A[c,r,mod(t+1)]*S[c,r,mod(t+1)]

	msg_closure!(msgs_up.s)
	msg_closure!(msgs_up.d)
	
	@tullio msgs_up.A[l,c,t] = P[k,mod(t-2)]*s[k,mod(t-2)]*A[k,l,mod(t-1)]*M[k,l,c,mod(t-1)]/msgs_up.s[l,mod(t-1)]
	@tullio msgs_up.S[l,c,t] = P[r,mod(t+1)]*d[r,mod(t+1)]*S[c,r,mod(t+1)]*M[l,c,r,t]/msgs_up.d[c,t]

	msg_closure!(msgs_up.A)
	msg_closure!(msgs_up.S)

	return msgs_up
end

function msg_gradient_descent!(msgs,msgs_up,w)
	# cw = 1-w
	# w2 = w^2
	# cw2 = 1-w2
	@tullio msgs.s[c,t] = (1-w)*msgs.s[c,t] + (w)*msgs_up.s[c,t]
	@tullio msgs.d[c,t] = (1-w)*msgs.d[c,t] + (w)*msgs_up.d[c,t]
	@tullio msgs.A[l,c,t] = (1-w^2)*msgs.A[l,c,t] + (w^2)*msgs_up.A[l,c,t]
	@tullio msgs.S[l,c,t] = (1-w^2)*msgs.S[l,c,t] + (w^2)*msgs_up.S[l,c,t]
	return msgs
end

function beliefs!(b,msgs,P)
	@tullio b[i,t] = P[i,t]*msgs.s[i,t]*msgs.d[i,t]
	msg_closure!(b)
	return b
end

beliefs(msgs,P) = beliefs!(copy(P),msgs,P)

function gbp_step!(msgs,msgs_up,w,M,P) 
	gbp_equations!(msgs_up,msgs,M,P)
	msg_gradient_descent!(msgs,msgs_up,w)
	return (msgs,msgs_up)
end

function gbp_step_random!(msgs,msgs_up,w,M,P)
	(;s,d,A,S) = msgs
	T = size(s,2)

	foreach(sample(1:T,T;replace=false)) do t
		tl = mod(t-1,axes(s,2))
		st_up = @view msgs_up.s[:,t]
		stl = @view s[:,tl]
		Ptl = @view P[:,tl]
		At = @view A[:,:,t]
		St = @view S[:,:,t]
		@tullio st_up[c] = Ptl[l]*stl[l]*At[l,c]*St[l,c]
		closure!(st_up)
		st = @view s[:,t]
		@tullio st[c] = (1-w)*st[c] + (w)*st_up[c]
		
		tr = mod(t+1,axes(s,2))
		dt_up = @view msgs_up.d[:,t]
		dtr = @view d[:,tr]
		Ptr = @view P[:,tr]
		Atr = @view A[:,:,tr]
		Str = @view S[:,:,tr]
		@tullio dt_up[c] = Ptr[r]*dtr[r]*Atr[c,r]*Str[c,r]
		closure!(dt_up)
		dt = @view d[:,t]
		@tullio dt[c] = (1-w)*dt[c] + (w)*dt_up[c]
	
		tll = mod(t-2,axes(s,2))
		At_up = @view msgs_up.A[:,:,t]
		Ptll = @view P[:,tll]
		stll = @view s[:,tll]
		Atl = @view A[:,:,tl]
		Mtl = @view M[:,:,:,tl]
		@tullio At_up[l,c] = Ptll[k]*stll[k]*Atl[k,l]*Mtl[k,l,c]/stl[l] # maybe wrong
		closure!(At_up)
		At = @view A[:,:,t]
		@tullio At[l,c] = (1-w^2)*At[l,c] + (w^2)*At_up[l,c]
	
		St_up = @view msgs_up.S[:,:,t]
		Mt = @view M[:,:,:,t]
		@tullio St_up[l,c] = Ptr[r]*dtr[r]*Str[c,r]*Mt[l,c,r]/dt[c]
		closure!(St_up)
		St = @view S[:,:,t]
		@tullio St[l,c] = (1-w^2)*St[l,c] + (w^2)*St_up[l,c]
	end

	return (msgs,msgs_up)
end

mmap(b) = mapreduce(first∘Tuple,vcat,last(findmax(b;dims=1)))
								
@fgenerator function decode!(b,M,P,w::Float64)
	M|>msg_closure!
	P|>msg_closure!
	msgs = msg_init(M)
	msgs_up = msg_init(M)
	while true
		gbp_step!(msgs,msgs_up,w,M,P)
		beliefs!(b,msgs,P)
		@yield (;Rs=mmap(b),beliefs=b)
	end
end

function decode(M,P,w::Float64)
	b = copy(P)
	decode!(b,M,P,w)
end

@fgenerator function decode_random!(b,M,P,w::Float64)
	M|>msg_closure!
	P|>msg_closure!
	msgs = msg_init(M)
	msgs_up = msg_init(M)
	while true
		gbp_step_random!(msgs,msgs_up,w,M,P)
		beliefs!(b,msgs,P)
		@yield (Rs=mmap(b),beliefs=copy(b))
	end
end

function decode_random(M,P,w)
	b = copy(P)
	decode_random!(b,M,P,w)
end

end # module GBPAlgorithm
