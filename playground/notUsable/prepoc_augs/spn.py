# Shuffle Pixel Noise: A random permutation of the pixels is chosen and then the same permutation is applied to all the images in the training set \cite{atya_non_2021}: 

# We can represent this as \begin{equation*}SPN(x) = (1 - \alpha ) * x + \alpha * {x_r}\tag{6}\end{equation*}, 
# where x is the original image, and xr is the image after shuffling pixels on the x, y axis. 
# This is applied with probability and magnitude: P=1, $\alpha$=1∗10−7.