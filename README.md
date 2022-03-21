# ECE271a-Statistical-Learning-I

**The goal is to segment the “cheetah” image (shown below in the left) into its two components, cheetah (foreground) and grass (background).

To formulate this as a pattern recognition problem, I need to decide on an observation space. Here I used the space of 8 × 8 image blocks, i.e. I view each image as a collection of 8 × 8 blocks. For each block I compute the discrete cosine transform (function dct2 on MATLAB) and obtain an array of 8 × 8 frequency coefficients. 

I then convert each 8 × 8 array into a 64 dimensional vector because it is easier to work with vectors than with arrays. The file **Zig-Zag Pattern.txt** contains the position (in the 1D vector) of each coefficient in the 8 × 8 array. The file **TrainingSamplesDCT 8.mat** contains a training set of vectors obtained from a similar image (stored as a matrix, each row is a training vector) for each of the classes. 

There are two matrices, TrainsampleDCT BG and TrainsampleDCT FG for foreground and background samples respectively.

To make the task of estimating the class conditional densities easier, I reduced each vector to a scalar. For each vector, I compute the index (position within the vector) of the coefficient that has the 2nd largest energy value (absolute value). This is my observation or feature X. By building an histogram of these indexes I obtained the class-conditionals for the two classes PX|Y (x|cheetah) and PX|Y (x|grass). The priors PY (cheetah) and PY (grass) should also be estimated from the training set.

Applying the Bayesian Decision Rule, in ImageProcessing file, I followed these steps to make classifications:
1) using the training data in **TrainingSamplesDCT 8.mat**, estimate the reasonable prior probabilities.
2) using the training data in **TrainingSamplesDCT 8.mat**, compute and plot the index histograms PX|Y (x|cheetah) and PX|Y (x|grass).
3) for each block in the image cheetah.bmp, compute the feature X (index of the DCT coefficient with 2nd greatest energy). Compute the state variable Y using the minimum probability of error rule based on the probabilities obtained in a) and b). Store the state in an array A. Using the commands imagesc and colormap(gray(255)) create a picture of that array.
4) The array A contains a mask that indicates which blocks contain grass and which contain the cheetah. Finally, I compare it with the ground truth provided in image cheetah mask.bmp (shown below on the right) and compute the probability of error of the algorithm.

In the second file, I'm going to assume that the class-conditional densities are multivariate Gaussians of 64 dimensions. I followed these steps to make classifications:
1) Using the training data in **TrainingSamplesDCT 8.mat** compute the histogram estimate of the prior PY (i), i ∈ {cheetah, grass}. Compute the maximum likelihood estimate for the prior probabilities. Compare the result with the estimates that obtained in ImageProcessing, interpret result.
2) Using the training data in **TrainingSamplesDCT 8.mat**, compute the maximum likelihood estimates for the parameters of the class conditional densities PX|Y (x|cheetah) and PX|Y (x|grass) under the Gaussian assumption. Select the best/worst 8 features for classification purposes.
3) Compute the Bayesian decision rule and classify the locations of the cheetah image using i) the 64-dimensional Gaussians, and ii) the 8-dimensional Gaussians associated with the best 8 features. Plot the classification masks and compute the probability of error by comparing with cheetah mask.bmp.
