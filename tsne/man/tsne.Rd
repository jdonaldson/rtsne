\name{tsne}
\Rdversion{1.1}
\alias{tsne}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
The t-SNE method for dimensionality reduction
}
\description{
Provides a simple function interface for specifying t-SNE dimensionality reduction on R matrices or "dist" objects.
}
\usage{
tsne(X, initial_config = NULL, k = 2, initial_dims = 30, perplexity = 30,
     max_iter = 1000, min_cost = 0, epoch_callback = NULL, whiten = TRUE,
     epoch=100)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
	The R matrix or "dist" object
}
  \item{initial_config}{
	an argument providing a matrix specifying the initial embedding for X.  See Details.
}
  \item{k}{
	the dimension of the resulting embedding.
}
  \item{initial_dims}{
	The number of dimensions to use in reduction method.
}
  \item{perplexity}{
	Perplexity parameter.  (optimal number of neighbors)
}
  \item{max_iter}{
	Maximum number of iterations to perform.
}
  \item{min_cost}{
	The minimum cost value (error) to halt iteration.
}
  \item{epoch_callback}{
	A callback function used after each epoch (an epoch here means a set number of iterations)
}
  \item{whiten}{
	A boolean value indicating whether the matrix data should be whitened.
}
  \item{epoch}{
	The number of iterations in between update messages.
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
	An R object containing a \emph{ydata} embedding matrix, as well as a the matrix of probabilities \emph{P}
}
\details{
	When the initial_config argument is specified, the algorithm will automatically enter the \emph{final momentum} stage.  This stage has less large scale adjustment to the embedding, and is intended for small scale tweaking of positioning.  This can greatly speed up the generation of embeddings for various similar X datasets, while also preserving overall embedding orientation.

}
\references{
 L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. \emph{Journal of Machine Learning Research} 9 (Nov) : 2579-2605, 2008.

 L.J.P. van der Maaten. Learning a Parametric Embedding by Preserving Local Structure. In \emph{Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics} (AISTATS), JMLR W&CP 5:384-391, 2009.
}
\author{
Justin Donaldson (jdonaldson@gmail.com)
}
%%\note{
%%  ~~further notes~~
%%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\link{dist}
}
\examples{\dontrun{
colors = rainbow(length(unique(iris$Species)))
names(colors) = unique(iris$Species)
ecb = function(x,y){ plot(x,t='n'); text(x,labels=iris$Species, col=colors[iris$Species]) }
tsne_iris = tsne(iris[,1:4], epoch_callback = ecb, perplexity=50)

# compare to PCA
dev.new()
pca_iris = princomp(iris[,1:4])$scores[,1:2]
plot(pca_iris, t='n')
text(pca_iris, labels=iris$Species,col=colors[iris$Species])
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
