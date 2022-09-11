using ILGPU.IR.Values;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static partial class MathEx
    {

		/// <summary>
		/// see wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
		/// 
		/// </summary>
		/// 
/*		def naive_covariance(data1, data2):

	n = len(data1)

	sum1 = sum(data1)

	sum2 = sum(data2)

	sum12 = sum([i1*i2 for i1, i2 in zip(data1, data2)])


	covariance = (sum12 - sum1* sum2 / n) / n
    return covariance



			def shifted_data_covariance(data_x, data_y) :

	n = len(data_x)
    if n< 2:
        return 0

	kx = data_x[0]

	ky = data_y[0]

	Ex = Ey = Exy = 0
    for ix, iy in zip(data_x, data_y):
        Ex += ix - kx
		Ey += iy - ky
		Exy += (ix - kx) * (iy - ky)
    return (Exy - Ex* Ey / n) / n


	void online_covariance(data1, data2):
{
			meanx = meany = C = n = 0

	for x, y in zip(data1, data2) :

		n += 1

		dx = x - meanx
		meanx += dx / n
		meany += (y - meany) / n
		C += dx * (y - meany)


	population_covar = C / n
# Bessel's correction for sample variance
	sample_covar = C / (n - 1)
	}









		Covariance
Very similar algorithms can be used to compute the covariance.

Naïve algorithm
The naïve algorithm is

{\displaystyle \operatorname {Cov
	}
	(X, Y)={\frac {\sum _ { i=1}^{n
}
x_
{ i}
y_
{ i}
-(\sum _ { i=1}
^{ n}
x_
{ i})(\sum _ { i=1}
^{ n}
y_
{ i})/ n}{ n}}.}{\displaystyle \operatorname { Cov} (X, Y) ={\frac {\sum _{ i = 1} ^{ n} x_{ i} y_{ i} -(\sum _{ i = 1} ^{ n} x_{ i})(\sum _{ i = 1} ^{ n} y_{ i})/ n} { n} }.}
For the algorithm above, one could use the following Python code:

def naive_covariance(data1, data2):
    n = len(data1)

	sum1 = sum(data1)

	sum2 = sum(data2)

	sum12 = sum([i1 * i2 for i1, i2 in zip(data1, data2) ])

	covariance = (sum12 - sum1 * sum2 / n) / n

	return covariance
With estimate of the mean
As for the variance, the covariance of two random variables is also shift-invariant, so given any two constant values {\displaystyle k_{x}}k_x and {\displaystyle k_ { y},}
{\displaystyle k_{ y},}
it can be written:

{\displaystyle \operatorname { Cov} (X, Y) =\operatorname { Cov} (X - k_{ x},Y - k_{ y})={\dfrac {\sum _{ i = 1} ^{ n} (x_{ i} -k_{ x})(y_{ i} -k_{ y})-(\sum _{ i = 1} ^{ n} (x_{ i} -k_{ x}))(\sum _{ i = 1} ^{ n} (y_{ i} -k_{ y}))/ n} { n} }.}
{\displaystyle \operatorname { Cov} (X, Y) =\operatorname { Cov} (X - k_{ x},Y - k_{ y})={\dfrac {\sum _{ i = 1} ^{ n} (x_{ i} -k_{ x})(y_{ i} -k_{ y})-(\sum _{ i = 1} ^{ n} (x_{ i} -k_{ x}))(\sum _{ i = 1} ^{ n} (y_{ i} -k_{ y}))/ n} { n} }.}
and again choosing a value inside the range of values will stabilize the formula against catastrophic cancellation as well as make it more robust against big sums. Taking the first value of each data set, the algorithm can be written as:

def shifted_data_covariance(data_x, data_y):
    n = len(data_x)

	if n < 2:
        return 0

	kx = data_x[0]

	ky = data_y[0]

	Ex = Ey = Exy = 0

	for ix, iy in zip(data_x, data_y):

		Ex += ix - kx

		Ey += iy - ky

		Exy += (ix - kx) * (iy - ky)

	return (Exy - Ex * Ey / n) / n
Two - pass
The two-pass algorithm first computes the sample means, and then the covariance:

{\displaystyle {\bar { x} }=\sum _{ i = 1} ^{ n} x_{ i}/ n}
{\displaystyle {\bar { x} }=\sum _{ i = 1} ^{ n} x_{ i}/ n}
{\displaystyle {\bar { y} }=\sum _{ i = 1} ^{ n} y_{ i}/ n}
{\displaystyle {\bar { y} }=\sum _{ i = 1} ^{ n} y_{ i}/ n}
{\displaystyle \operatorname { Cov} (X, Y) ={\frac {\sum _{ i = 1} ^{ n} (x_{ i} -{\bar { x} })(y_{ i} -{\bar { y} })} { n} }.}
{\displaystyle \operatorname { Cov} (X, Y) ={\frac {\sum _{ i = 1} ^{ n} (x_{ i} -{\bar { x} })(y_{ i} -{\bar { y} })} { n} }.}
The two-pass algorithm may be written as:

def two_pass_covariance(data1, data2):
    n = len(data1)

	mean1 = sum(data1) / n

	mean2 = sum(data2) / n


	covariance = 0

	for i1, i2 in zip(data1, data2):

		a = i1 - mean1

		b = i2 - mean2

		covariance += a * b / n

	return covariance
A slightly more accurate compensated version performs the full naive algorithm on the residuals. The final sums {\textstyle \sum _{i}x_
{ i}}{\textstyle \sum_i x_i}
and
{\textstyle \sum _{ i} y_{ i} }
{\textstyle \sum _{ i} y_{ i} }
should be zero, but the second pass compensates for any small error.

Online
A stable one-pass algorithm exists, similar to the online algorithm for computing the variance, that computes co-moment {\textstyle C_{n}=\sum _ { i=1}
^{ n} (x_{ i}
-{\bar { x} }
_
{ n})(y_{ i}
-{\bar { y} }
_
{ n})}{\textstyle C_{ n}=\sum _{ i = 1} ^{ n} (x_{ i} -{\bar { x} } _{ n})(y_{ i} -{\bar { y} } _{ n})}:

{\displaystyle {\begin{ alignedat} { 2} {\bar { x} } _{ n}&={\bar { x} } _{ n - 1} &\,+\,&{\frac { x_{ n} -{\bar { x} } _{ n - 1} } { n} }\\[5pt]{\bar { y} } _{ n}&={\bar { y} } _{ n - 1} &\,+\,&{\frac { y_{ n} -{\bar { y} } _{ n - 1} } { n} }\\[5pt]C_{ n}&= C_{ n - 1} &\,+\,&(x_{ n} -{\bar { x} } _{ n})(y_{ n} -{\bar { y} } _{ n - 1})\\[5pt]&= C_{ n - 1} &\,+\,&(x_{ n} -{\bar { x} } _{ n - 1})(y_{ n} -{\bar { y} } _{ n})\end{ alignedat} } }
{\displaystyle {\begin{ alignedat} { 2} {\bar { x} } _{ n}&={\bar { x} } _{ n - 1} &\,+\,&{\frac { x_{ n} -{\bar { x} } _{ n - 1} } { n} }\\[5pt]{\bar { y} } _{ n}&={\bar { y} } _{ n - 1} &\,+\,&{\frac { y_{ n} -{\bar { y} } _{ n - 1} } { n} }\\[5pt]C_{ n}&= C_{ n - 1} &\,+\,&(x_{ n} -{\bar { x} } _{ n})(y_{ n} -{\bar { y} } _{ n - 1})\\[5pt]&= C_{ n - 1} &\,+\,&(x_{ n} -{\bar { x} } _{ n - 1})(y_{ n} -{\bar { y} } _{ n})\end{ alignedat} } }
The apparent asymmetry in that last equation is due to the fact that {\textstyle (x_{n}-{\bar { x} }
_
{ n})={\frac { n - 1} { n} } (x_{ n}
-{\bar { x} }
_
{ n - 1})}{\textstyle(x_{ n} -{\bar { x} } _{ n})={\frac { n - 1} { n} } (x_{ n} -{\bar { x} } _{ n - 1})}, so both update terms are equal to {\textstyle {\frac {n-1}{ n}}(x_{ n}
-{\bar { x} }
_
{ n - 1})(y_{ n}
-{\bar { y} }
_
{ n - 1})}{\textstyle {\frac { n - 1} { n} } (x_{ n} -{\bar { x} } _{ n - 1})(y_{ n} -{\bar { y} } _{ n - 1})}. Even greater accuracy can be achieved by first computing the means, then using the stable one - pass algorithm on the residuals.

Thus the covariance can be computed as

{\displaystyle {\begin{aligned}\operatorname
{ Cov}
_
{ N}
(X, Y) ={\frac { C_{ N} } { N} }&={\frac {\operatorname { Cov} _{ N - 1} (X, Y)\cdot(N - 1) + (x_{ n} -{\bar { x} } _{ n})(y_{ n} -{\bar { y} } _{ n - 1})} { N} }\\&={\frac {\operatorname { Cov} _{ N - 1} (X, Y)\cdot(N - 1) + (x_{ n} -{\bar { x} } _{ n - 1})(y_{ n} -{\bar { y} } _{ n})} { N} }\\&={\frac {\operatorname { Cov} _{ N - 1} (X, Y)\cdot(N - 1) +{\frac { N - 1} { N} } (x_{ n} -{\bar { x} } _{ n - 1})(y_{ n} -{\bar { y} } _{ n - 1})} { N} }\\&={\frac {\operatorname { Cov} _{ N - 1} (X, Y)\cdot(N - 1) +{\frac { N} { N - 1} } (x_{ n} -{\bar { x} } _{ n})(y_{ n} -{\bar { y} } _{ n})} { N} }.\end
{ aligned}}}{\displaystyle {\begin{ aligned}\operatorname { Cov}
		_{ N}
		(X, Y) ={\frac { C_{ N} } { N} }&={\frac {\operatorname { Cov} _{ N - 1} (X, Y)\cdot(N - 1) + (x_{ n} -{\bar { x} } _{ n})(y_{ n} -{\bar { y} } _{ n - 1})} { N} }\\&={\frac {\operatorname {Cov} _{N-1}(X,Y)\cdot (N-1)+(x_{n}-{\bar {x}}_{n-1})(y_{n}-{\bar {y}}_{n})}{N}}\\&={\frac {\operatorname {Cov} _{N-1}(X,Y)\cdot (N-1)+{\frac {N-1}{N}}(x_{n}-{\bar {x}}_{n-1})(y_{n}-{\bar {y}}_{n-1})}{N}}\\&={\frac {\operatorname {Cov} _{N-1}(X,Y)\cdot (N-1)+{\frac {N}{N-1}}(x_{n}-{\bar {x}}_{n})(y_{n}-{\bar {y}}_{n})}{N}}.\end{aligned}}}
def online_covariance(data1, data2):
    meanx = meany = C = n = 0
    for x, y in zip(data1, data2):
        n += 1
        dx = x - meanx
        meanx += dx / n
        meany += (y - meany) / n
        C += dx * (y - meany)

    population_covar = C / n
    # Bessel's correction for sample variance
    sample_covar = C / (n - 1)
A small modification can also be made to compute the weighted covariance:

def online_weighted_covariance(data1, data2, data3):
    meanx = meany = 0
    wsum = wsum2 = 0
    C = 0
    for x, y, w in zip(data1, data2, data3):
        wsum += w
        wsum2 += w * w
        dx = x - meanx
        meanx += (w / wsum) * dx
        meany += (w / wsum) * (y - meany)
        C += w * dx * (y - meany)

    population_covar = C / wsum
    # Bessel's correction for sample variance
    # Frequency weights
    sample_frequency_covar = C / (wsum - 1)
    # Reliability weights
    sample_reliability_covar = C / (wsum - wsum2 / wsum)
Likewise, there is a formula for combining the covariances of two sets that can be used to parallelize the computation:[3]

{\displaystyle C_{X}=C_{A}+C_{B}+({\bar {x}}_{A}-{\bar {x}}_{B})({\bar {y}}_{A}-{\bar {y}}_{B})\cdot {\frac {n_{A}n_{B}}{n_{X}}}.}{\displaystyle C_{X}=C_{A}+C_{B}+({\bar {x}}_{A}-{\bar {x}}_{B})({\bar {y}}_{A}-{\bar {y}}_{B})\cdot {\frac {n_{A}n_{B}}{n_{X}}}.}
    */}
}
