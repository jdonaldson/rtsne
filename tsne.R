tsne<-function(X,initial_config = Z, k=2, initial_dims=30, perplexity=30, max_iter = 1000, min_cost=0, epoch_callback=NULL,whiten=TRUE){
	message(1)
	if (class(X) == 'dist') { 
		n = attr(X,'Size')
		X = X/sum(X)
		X = as.matrix(X)
		}
	else 	{
		if (whiten) X<-.whiten(X,n.comp=initial_dims)
		n = dim(X)[1]
	}

	momentum = .5
	final_momentum = .8
	mom_switch_iter = 250

	epsilon = 500
	min_gain = .01
	

	P = .x2p(X,perplexity, 1e-5)$P
	eps = 2^(-52) # typical machine precision
	P[is.nan(P)]<-eps
	P = .5 * (P + t(P))
	P = P / sum(P)
	P[P < eps]<-eps
	P = P * 4
	if (!is.null(initial_config)) { 
		ydata = initial_config
	} else {
		ydata = matrix(rnorm(k * nrow(X)),nrow(X))
	}
	y_grads =  matrix(0,dim(ydata)[1],dim(ydata)[2])
	y_incs =  matrix(0,dim(ydata)[1],dim(ydata)[2])
	gains = matrix(1,dim(ydata)[1],dim(ydata)[2])
	
	for (iter in 1:max_iter){
		sum_ydata = apply(ydata^2, 1, sum)
		num =  1/(1 + sum_ydata +    sweep(-2 * ydata %*% t(ydata),2, -t(sum_ydata))) 
		diag(num)=0
		Q = num / sum(num)
		if (any(is.nan(num))) message ('NaN in grad. descent')
		Q[Q < eps] = eps
		stiffnesses = 4 * (P-Q) * num
		for (i in 1:n){
			y_grads[i,] = apply(sweep(-ydata, 2, -ydata[i,]) * stiffnesses[,i],2,sum)
		}
		
		gains = (gains + .2) * abs(sign(y_grads) != sign(y_incs)) 
				+ gains * .8 * abs(sign(y_grads) == sign(y_incs))		
		gains[gains < min_gain] = min_gain
		y_incs = momentum * y_incs - epsilon * (gains * y_grads)
		ydata = ydata + y_incs
		y_data = sweep(ydata,2,apply(ydata,2,mean))
		if (iter == mom_switch_iter) momentum = final_momentum
		
		if (iter == 100) P = P/4
		
		if (iter %% 10 == 0) { # epoch
			cost =  sum(apply(P * log((P+eps)/(Q+eps)),1,sum))
			message("Epoch: Iteration #",iter," error is: ",cost)
			if (cost < min_cost) break
			if (!is.null(epoch_callback)) epoch_callback(yata, P)
		}
	
		
	}
	r = {}
	r$ydata = ydata
	r$P = P
	r
	
}


.x2p<-function(X,perplexity = 15,tol = 1e-5){
	if (class(X) == 'dist') {
		n = attr(X,'Size')
		D = X
	} else{
		n = nrow(X)
		D = dist(X)^2 # remove the square once this is resolved
	}
		
	D = as.matrix(D)

	
	P = matrix(0, n, n )		
	beta = rep(1, n)
	logU = log(perplexity)
	
	for (i in 1:n){
		betamin = -Inf
		betamax = Inf
		Di = D[i, -i]
		mDi <<-Di
		hbeta= .Hbeta(Di, beta[i])
		H = hbeta$H; 
		thisP = hbeta$P
		Hdiff = H - logU;
		tries = 0;

		while(abs(Hdiff) > tol && tries < 50){
			if (Hdiff > 0){
				betamin = beta[i]
				if (is.infinite(betamax)) beta[i] = beta[i] * 2
				else beta[i] = (beta[i] + betamax)/2
			} else{
				betamax = beta[i]
				if (is.infinite(betamin))  beta[i] = beta[i]/ 2
				else beta[i] = ( beta[i] + betamin) / 2
			}
			
			hbeta = .Hbeta(Di, beta[i])
			H = hbeta$H
			thisP = hbeta$P
			Hdiff = H - logU
			tries = tries + 1
		}	
			P[i,-i]  = thisP	
	}	
	
	r = {}
	r$P = P
	r$beta = beta
	sigma = sqrt(1/beta)
	
	message('sigma summary: ', paste(names(summary(sigma)),':',summary(sigma),'|',collapse=''))

	r 
}


.Hbeta<-function(D, beta){
	P = exp(-D * beta)
	sumP = sum(P)
	H = log(sumP) + beta * sum(D * P) /sumP
	P = P/sumP
	r = {}
	r$H = H
	r$P = P
	r
}

# Whitens matrix-conformal data using scaled covariance matrix svd technique
.whiten<-function(X, row.norm=FALSE, verbose=FALSE, n.comp=ncol(X))
{  
	n.comp; # forces an eval/save of n.comp
	if (verbose) message("Centering")
   n = nrow(X)
	p = ncol(X)
	X <- scale(X, scale = FALSE)
   X <- if (row.norm) 
       t(scale(X, scale = row.norm))
   else t(X)

   if (verbose) message("Whitening")
   V <- X %*% t(X)/n
   s <- La.svd(V)
   D <- diag(c(1/sqrt(s$d)))
   K <- D %*% t(s$u)
   K <- matrix(K[1:n.comp, ], n.comp, p)
   X = t(K %*% X)
	X
}

