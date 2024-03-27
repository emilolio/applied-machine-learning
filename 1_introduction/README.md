
2. Load the `digit` data set from `sklearn`
	1. It contains a data set of handwritten digits, called MNIST
	2. The set is 1797 digits
	3. Each cell is colored black/white, i.e. some intensity slider, with a range of `0` to `16`
	4. The ordering of the data, is it necessary?
3. UMAP reducer
	1. UMAP reduces the dimensionality of the data set, giving us the ability to visualize data in 2D space

4. Calculating *cosine similarity*
				Average of images
2. Calculate the $L_2$-norm for the raw data
	2. Where $L_2=\sqrt{\sum_{i=0}^{N}{v_i^{2}}}$
3. Calculate the cosine similarity, formula 
$$ 
\begin{align}
cos(\theta)=\frac{A*B}{||A|| ||B||} \\ \\
\theta = arccos(\frac{A*B}{||A|| ||B||})
\end{align}
$$
 
