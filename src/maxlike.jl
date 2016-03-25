

module maxlike

	using Distributions, Optim, PyPlot, DataFrames, Debug

	"""
    `input(prompt::AbstractString="")`
  
    Read a string from STDIN. The trailing newline is stripped.
  
    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    function input(prompt::AbstractString="")
        print(prompt)
        return chomp(readline())
    end

    export runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm
	# true coeff vector, number of obs, data matrix X (Nxk), response vector y (binary), and a type of parametric distribution; i.e. the standard normal in our case.

	function makeData(n=10000, usual=true)
		beta = [ 1.0; 1.5; -0.5 ]
		mu = [0.0,0.0]
 		sig = diagm([1.0;1.0])
 		X2X3=rand(MvNormal(mu,sig),n)
 		X1=ones(n)
 		X=hcat(X1,transpose(X2X3))
 		dist = Normal()
 		if usual
 			v=rand(Normal(),n)
 			Ystar=X*beta+v
 			Y=zeros(n)
 			for i in 1:n
 				if Ystar[i] > 0.0
 					Y[i]=1.0
 				end
 			end
 		else
 			prob = cdf(dist,X*beta)
			Y=zeros(n)
 			for i in 1:n
 				 Y[i]=rand(Bernoulli(prob[i]),1)[1]
 			end
 		end
 		return Dict("y"=>Y,"x"=>X, "n"=>n,"beta"=>beta,"dist"=>dist)
 	end

	# we need to use the dictionnary to store makeData's values:

	d = makeData()
	global d


	# log likelihood function at x

	function loglik(beta::Vector,d=d)
		
		prob1=cdf(d["dist"],d["x"]*beta)

		Z=zeros(d["n"])
		for i in 1:length(d["y"])
			if 	d["y"][i]==1.0
					Z+=log(prob1[i])
			else
					Z+=log(1-prob1[i])
			end
		end
		return -sum(Z)

	end


	function plotLike(d=d)

		beta1=linspace(-0.5,2.5,d["n"])
		beta2=linspace(0,3,d["n"])
		beta3=linspace(-1,2,d["n"])
		y1=zeros(d["n"])
		y2=zeros(d["n"])
		y3=zeros(d["n"])

		for i in 1:300

			y1[i]=-loglik([beta1[i];1.5;0.5],d)
			y2[i]=-loglik([1;beta2[i];0.5],d)
			y3[i]=-loglik([1;1.5;beta3[i]],d)

		end

		figure("Betas varying")
 		subplot(311)
 		plot(collect(beta1),y1)
 		title("beta1")
 	subplot(312)
 		plot(collect(beta2),y2)
 	title("beta2")
 		subplot(313)
 		plot(collect(beta3),y3)
 		title("beta3")
	end


	# gradient of the likelihood at x

	function grad!(beta::Vector,storage::Vector,d=d)
	
	for i in 1:length(storage)

 			Z=zeros(d["n"])

 			for j in 1:length(d["y"])
 				
 			Z=d["y"][i]*d["x"][j,i]*pdf(d["dist"],d["x"][j,:]*beta)/cdf(d["dist"],d["x"][j,:]*beta)+(1-d["y"][i])-d["x"][j,i]*pdf(d["dist"],d["x"][j,:]*beta)/(1-cdf(d["dist"],d["x"][j,:]*beta))
 			
 			end
 		
 			storage[i]=-sum(Z)
 		end
	end


	# hessian of the likelihood at x
	function hessian!(beta::Vector,storage::Matrix,d=d)

	storage[:,:]=0

	for i in 1:length(d["y"])

	dens2=pdf(d["dist"],d["x"][line=i]*beta)
	prob2=cdf(d["dist"],d["x"][line=i]*beta)

		if d["y"][i]==1.0
 				storage[1,1]+=dens2*d["x"][i,1]*d["x"][i,1]*((dens2.+(d["x"][i,:]*beta)[1].*(prob2))/(prob2^2))
 				storage[2,1]+=dens2*d["x"][i,2]*d["x"][i,1]*((dens2.+(d["x"][i,:]*beta)[1].*(prob2))/(prob2^2))
 				storage[3,1]+=dens2*d["x"][i,3]*d["x"][i,1]*((dens2.+(d["x"][i,:]*beta)[1].*(prob2))/(prob2^2))
 				storage[2,2]+=dens2*d["x"][i,2]*d["x"][i,2]*((dens2.+(d["x"][i,:]*beta)[1].*(prob2))/(prob2^2))
 				storage[3,2]+=dens2*d["x"][i,3]*d["x"][i,2]*((dens2.+(d["x"][i,:]*beta)[1].*(prob2))/(prob2^2))
 				storage[3,3]+=dens2*d["x"][i,3]*d["x"][i,3]*((dens2.+(d["x"][i,:]*beta)[1].*(prob2))/(prob2^2))
 			else
 				storage[1,1]+=dens2*d["x"][i,1]*d["x"][i,1]*((dens2-(d["x"][i,:]*beta)[1].*(1-prob2))/((1-prob2)^2))
 				storage[2,1]+=dens2*d["x"][i,2]*d["x"][i,1]*((dens2-(d["x"][i,:]*beta)[1].*(1-prob2))/((1-prob2)^2))
 				storage[3,1]+=dens2*d["x"][i,3]*d["x"][i,1]*((dens2-(d["x"][i,:]*beta)[1].*(1-prob2))/((1-prob2)^2))
 				storage[2,2]+=dens2*d["x"][i,2]*d["x"][i,2]*((dens2-(d["x"][i,:]*beta)[1].*(1-prob2))/((1-prob2)^2))
 				storage[3,2]+=dens2*d["x"][i,3]*d["x"][i,2]*((dens2-(d["x"][i,:]*beta)[1].*(1-prob2))/((1-prob2)^2))
 				storage[3,3]+=dens2*d["x"][i,3]*d["x"][i,3]*((dens2-(d["x"][i,:]*beta)[1].*(1-prob2))/((1-prob2)^2))
 			end
 		end
 		#Matrix' values are symmetric
 		storage[1,2]=storage[2,1]
 		storage[2,3]=storage[3,2]
 		storage[1,3]=storage[3,1]
 	end

	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(beta::Vector,d)
		storage=zeros(3,3)
		hessian!(beta::Vector,storage::Matrix,d)
		return inv(storage)
	end

	"""
	standard errors
	"""
	function se(beta::Vector,d::Dict)
		return sqrt(diag(inv_observedInfo(beta,d)))
	end

	# function that maximizes the log likelihood without the gradient

	# with a call to `optimize` and returns the result
	
	function maximize_like(x0=[0.8,1.0,-0.1],method=:nelder_mead)
	logl(t)=loglik(t,makeData())
	optimize(loglik,x0,method=:nelder_mead,iterations=10000)
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result

	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=:bfgs)
	logl(t) = loglik(t,makeData())
 	grad(t,s) = grad!(t,s,makeData())
 	optimize(logl, grad, x0, method = :bfgs,iterations=10000)
	end

	# function that maximizes the log likelihood with the gradient
	# and hessian with a call to `optimize` and returns the result

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],method=:newton)
	logl(t) = loglik(t,makeData())
 	grad(t,s) = grad!(t,s,makeData())
 	hess(t,s) = hessian!(t,s,makeData())
	optimize(loglik, grad, hess,x0,method=:newton,iterations=10000)
	end

	# function that maximizes the log likelihood with the gradient
	# and computes the standard errors for the estimates
	# should return a dataframe with 3 rows
	# first column should be parameter names
	# second column "Estimates"
	# third column "StandardErrors"

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],method=:bfgs)
	logl(t) = loglik(t,makeData())
 	grad(t,s) = grad!(t,s,makeData())
 	optse = optimize(logl,grad,x0, method=:bfgs,iterations=10000)
 	standard=se(optse.minimum,makeData())
 	results=hcar(["beta1";"beta2";"beta3"],optse.minimum,standard)
 	return results
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the true value.





	function runAll()
		plotLike()
		m1 = maximize_like()
		m2 = maximize_like_grad()
		m3 = maximize_like_grad_hess()
		m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like: $(m1.minimum)")
		println("maximize_like_grad: $(m2.minimum)")
		println("maximize_like_grad_hess: $(m3.minimum)")
		println("maximize_like_grad_se: $m4")
		println("")
		println("running tests:")
		include("test/runtests.jl")
		println("")
		ok = input("enter y to close this session.")
		if ok == "y"
			quit()
		end
	end
end




