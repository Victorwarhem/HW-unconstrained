

using FactCheck
 using GLM
 using maxlike
 using DataFrames
 
 context("basics") do
 	data=maxlike.makeData()
 	facts("Test Data Construction") do
 		@fact typeof(d) --> Dict{ASCIIString,Any}
 		@fact typeof(d["x"])--> Array{Float64,2}
 		@fact maximum(d["x"][:,1]) --> 1.0
 		@fact minimum(d["x"][:,1]) --> 1.0
 	end
 
 	facts("Test Return value of likelihood") do
 		@fact maxlike.loglik([100.0,100.0,100.0],d) <1e-8 ->true
 		@fact maxlike.loglik(d["beta"],d) - maxlike.loglik([randn()+d["beta"][1];randn()+d["beta"][2];randn()+d["beta"][3]],d) >=0 -->true
 
 	end
 
 	facts("Test return value of gradient") do
 		storage=zeros(3)
 		@fact maxlike.grad!([randn()+d["beta"][1];randn()
 		d["beta"][2];randn()+d["beta"][3]],storage,d) --> nothing
 		@fact storage != zeros(3) -->true
 	end
 end
 
 context("test maximization results") do
 	d=maxlike.makeData()
 	facts("maximize returns approximate result") do
 		result=maxlike.maximize_like()
 		for i in 1:length(d["beta"])
 			@fact abs(result.minimum[i]-d["beta"][i])<0.1-->true
 		end
 
 	end
 
 	facts("maximize_grad returns accurate result") do
 		result=maxlike.maximize_like_grad()
 		for i in 1:length(d["beta"])
 			@fact abs(result.minimum[i]-d["beta"][i])<0.1->true
 		end
 	end
 
 	facts("maximize_grad_hess returns accurate result") do
 		result=maxlike.maximize_like_grad_hess()
 		for i in 1:length(d["beta"])
 			@fact abs(result.minimum[i]-d["beta"][i])<0.1-->true
 		end
 	end
 
 end
 
 context("test against GLM") do
 	srand(20160324)
 	d=maxlike.makeData()
 	estim=maxlike.maximize_like_grad_se()
 	d_df=DataFrame(y=d["y"],d1=d["x"][:,1],d2=d["x"][:,2],d3=d["x"][:,3])
 	Probit = glm(y~d2+d3,d_df,Binomial(),ProbitLink())
 	facts("estimates vs GLM") do
 	diff_coeff=abs(coef(Probit)-estim[:optse.minimum])
 		for i in 1:length(diff_coeff)
 			@fact diff_coeff[i] < 1e-5 -->true
 		end
 
  	end
 
 	facts("standard errors vs GLM") do
 		diff_se=abs(stderr(Probit)-estim[:standard])
 		for i in 1:length(diff_se)
 			@fact diff_se[i]<1e-4 -->true
 		end
 	end
 
 end



end

