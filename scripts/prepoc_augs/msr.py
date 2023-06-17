# Mixed Structure Regularisation: corrupt the input data by randomly adding different brain images from BraTS training set, as described by \cite{atya_non_2021}. 

# We can represent this data augmentation with the function: \begin{equation*}MSR(x) = (1 - \alpha ) * x + \alpha * {x_r}\tag{5}\end{equation*}. 
# where x is the original image, and xr is a randomly selected image. 
# This is applied with probability and magnitude of P=0.5, $\alpha$=1∗10−4.
