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
reset_msg!(m) = (fill!(m,1.0)|>msg_closure!; nothing)

struct GBPMessages
	s
	d
	A
	S
end

GBPMessages(N,T) = GBPMessages(msg_init(N,T),msg_init(N,T),msg_init(N,N,T),msg_init(N,N,T))

reset_msg!(m::GBPMessages) = (reset_msg!(m.s);reset_msg!(m.d);reset_msg!(m.A);reset_msg!(m.S); nothing)

abstract type AbstractGBPDecoder end

struct GBPDecoder <: AbstractGBPDecoder
	msgs::GBPMessages
	msgs_up::GBPMessages
	w::Float64
end

GBPDecoder(N::Int,T::Int,w::Float64) = GBPDecoder(GBPMessages(N,T),GBPMessages(N,T),w)

reset_msg!(X::GBPDecoder) = (reset_msg!(X.msgs); reset_msg!(X.msgs_up); nothing)

@fgenerator function (X::GBPDecoder)(F)
	while true
		gbp_step!(X,F)
		beliefs!(F,X)
		@yield F
	end
end

struct Factors
	M
	P
	b
	Rs
end

Factors(N,T) = Factors(msg_init(N,N,N,T),msg_init(N,T),msg_init(N,T),fill(-1,T))

mmap!(ind,b) = map!(argmax,ind,eachcol(b))
mmap(b) = (ind=Array{Int}(undef,size(b,2)); mmap!(ind,b))

function beliefs!(F,X)
	@tullio F.b[i,t] = F.P[i,t]*X.msgs.s[i,t]*X.msgs.d[i,t]
	msg_closure!(F.b)
	mmap!(F.Rs,F.b)
	return F
end

function gbp_equations!(X,F)
	(;s,d,A,S) = X.msgs
	(;M,P) = F

	@tullio X.msgs_up.s[c,t] = P[l,mod(t-1)]*s[l,mod(t-1)]*A[l,c,t]*S[l,c,t]
	@tullio X.msgs_up.d[c,t] = P[r,mod(t+1)]*d[r,mod(t+1)]*A[c,r,mod(t+1)]*S[c,r,mod(t+1)]

	msg_closure!(X.msgs_up.s)
	msg_closure!(X.msgs_up.d)
	
	@tullio X.msgs_up.A[l,c,t] = P[k,mod(t-2)]*s[k,mod(t-2)]*A[k,l,mod(t-1)]*M[k,l,c,mod(t-1)]/X.msgs_up.s[l,mod(t-1)]
	@tullio X.msgs_up.S[l,c,t] = P[r,mod(t+1)]*d[r,mod(t+1)]*S[c,r,mod(t+1)]*M[l,c,r,t]/X.msgs_up.d[c,t]

	msg_closure!(X.msgs_up.A)
	msg_closure!(X.msgs_up.S)

	return X
end

function msg_gradient_descent!(X)
	w = X.w
	@tullio X.msgs.s[c,t] = (1-w)*X.msgs.s[c,t] + (w)*X.msgs_up.s[c,t]
	@tullio X.msgs.d[c,t] = (1-w)*X.msgs.d[c,t] + (w)*X.msgs_up.d[c,t]
	@tullio X.msgs.A[l,c,t] = (1-w^2)*X.msgs.A[l,c,t] + (w^2)*X.msgs_up.A[l,c,t]
	@tullio X.msgs.S[l,c,t] = (1-w^2)*X.msgs.S[l,c,t] + (w^2)*X.msgs_up.S[l,c,t]
	return X
end

function gbp_step!(X,F) 
	gbp_equations!(X,F)
	msg_gradient_descent!(X)
	return X
end

uniform_prior!(F) = (fill!(F.P,1.0)|>msg_closure!; F)

function collapse_prior!(F,collapsed::Pair...)
	uniform_prior!(F)
	foreach(collapsed) do (t,q)
		fill!(view(F.P,:,t),0.0)
		fill!(view(F.P,q,t),1.0)
	end
	return F
end
collapse_prior!(F,collapsed::Int...) = collapse_prior!(F,map(splat(Pair),enumerate(collapsed))...)
collapse_prior!(F) = uniform_prior!(F)

end # module GBPAlgorithm