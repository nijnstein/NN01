using ILGPU.IR.Values;
using System.Runtime.CompilerServices;

namespace NSS
{
    /// <summary>
    /// SimpleRNG is a simple random number generator based on 
    /// George Marsaglia's MWC (multiply with carry) generator.
    /// Although it is very simple, it passes Marsaglia's DIEHARD
    /// series of random number generator tests.
    /// 
    /// Written by John D. Cook 
    /// http://www.johndcook.com
    /// 
    /// changed to non-static + float support 
    /// </summary>
    public class SimpleRNG
    {
        private uint m_w;
        private uint m_z;

        public SimpleRNG()
        {
            // These values are not magical, just the default values Marsaglia used.
            // Any pair of unsigned integers should be fine.
            m_w = 521288629;
            m_z = 362436069;
        }

        public SimpleRNG(uint w, uint z)
        {
            m_w = w;
            m_z = z;
        }

        public SimpleRNG(uint u) : this()
        {
            m_w = u;
        }
        public SimpleRNG(DateTime time) : this()
        {
            SetSeedFromSystemTime(time); 
        }


        // The random generator seed can be set three ways:
        // 1) specifying two non-zero unsigned integers
        // 2) specifying one non-zero unsigned integer and taking a default value for the second
        // 3) setting the seed from the system time

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetSeed(uint u, uint v)
        {
            if (u != 0) m_w = u;
            if (v != 0) m_z = v;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetSeed(uint u)
        {
            m_w = u;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetSeedFromSystemTime(DateTime time)
        {
            long x = time.ToFileTime();
            SetSeed((uint)(x >> 16), (uint)(x % 4294967296));
        }

        // Produce a uniform random sample from the open interval (0, 1).
        // The method will not return either end point.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetUniform()
        {
            // 0 <= u < 2^32
            uint u = GetUint();
            // The magic number below is 1/(2^32 + 2).
            // The result is strictly between 0 and 1.
            return (u + 1.0) * 2.328306435454494e-10;
        }

        // Produce a uniform random sample from the open interval (0, 1).
        // The method will not return either end point.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetUniformSingle()
        {
            // 0 <= u < 2^32
            uint u = GetUint();
            // The magic number below is 1/(2^32 + 2).
            // The result is strictly between 0 and 1.
            return (u + 1.0f) * 2.328306435454494e-10f;
        }

        // This is the heart of the generator.
        // It uses George Marsaglia's MWC algorithm to produce an unsigned integer.
        // See http://www.bobwheeler.com/statistics/Password/MarsagliaPost.txt
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private uint GetUint()
        {
            m_z = 36969 * (m_z & 65535) + (m_z >> 16);
            m_w = 18000 * (m_w & 65535) + (m_w >> 16);
            return (m_z << 16) + m_w;
        }

        // 0..max exluding max 
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetInt(int max)
        {
            return MathEx.FloorToInt(GetUniformSingle() * max);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetInt(int min, int max)
        {
            return MathEx.FloorToInt(GetUniformSingle() * (max - min)) + min;
        }

        // Get normal (Gaussian) random sample with mean 0 and standard deviation 1
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetNormal()
        {
            // Use Box-Muller algorithm
            double u1 = GetUniform();
            double u2 = GetUniform();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            return r * Math.Sin(theta);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetNormalSingle()
        {
            // Use Box-Muller algorithm
            float u1 = GetUniformSingle();
            float u2 = GetUniformSingle();
            float r = MathF.Sqrt(-2F * MathF.Log(u1));
            float theta = 2F * MathF.PI * u2;
            return r * MathF.Sin(theta);
        }

        // Get normal (Gaussian) random sample with specified mean and standard deviation
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetNormal(double mean, double standardDeviation)
        {
            if (standardDeviation <= 0.0)
            {
                string msg = string.Format("Shape must be positive. Received {0}.", standardDeviation);
                throw new ArgumentOutOfRangeException(msg);
            }
            return mean + standardDeviation * GetNormal();
        }

        // Get normal (Gaussian) random sample with specified mean and standard deviation
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetNormalSingle(float mean, float standardDeviation)
        {              
            if (standardDeviation <= 0.0)
            {
                string msg = string.Format("Shape must be positive. Received {0}.", standardDeviation);
                throw new ArgumentOutOfRangeException(msg);
            }
            return mean + standardDeviation * GetNormalSingle();
        }

        // Get exponential random sample with mean 1
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetExponential()
        {
            return -Math.Log(GetUniform());
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetExponentialSingle()
        {
            return -MathF.Log(GetUniformSingle());
        }

        // Get exponential random sample with specified mean
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetExponential(double mean)
        {
            if (mean <= 0.0)
            {
                string msg = string.Format("Mean must be positive. Received {0}.", mean);
                throw new ArgumentOutOfRangeException(msg);
            }
            return mean * GetExponential();
        }

        // Get exponential random sample with specified mean
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetExponential(float mean)
        {
            if (mean <= 0.0)
            {
                string msg = string.Format("Mean must be positive. Received {0}.", mean);
                throw new ArgumentOutOfRangeException(msg);
            }
            return mean * GetExponentialSingle();
        }

        public double GetGamma(double shape, double scale)
        {
            // Implementation based on "A Simple Method for Generating Gamma Variables"
            // by George Marsaglia and Wai Wan Tsang.  ACM Transactions on Mathematical Software
            // Vol 26, No 3, September 2000, pages 363-372.

            double d, c, x, xsquared, v, u;

            if (shape >= 1.0)
            {
                d = shape - 1.0 / 3.0;
                c = 1.0 / Math.Sqrt(9.0 * d);
                for (; ; )
                {
                    do
                    {
                        x = GetNormal();
                        v = 1.0 + c * x;
                    }
                    while (v <= 0.0);
                    v = v * v * v;
                    u = GetUniform();
                    xsquared = x * x;
                    if (u < 1.0 - .0331 * xsquared * xsquared || Math.Log(u) < 0.5 * xsquared + d * (1.0 - v + Math.Log(v)))
                        return scale * d * v;
                }
            }
            else if (shape <= 0.0)
            {
                string msg = string.Format("Shape must be positive. Received {0}.", shape);
                throw new ArgumentOutOfRangeException(msg);
            }
            else
            {
                double g = GetGamma(shape + 1.0, 1.0);
                double w = GetUniform();
                return scale * g * Math.Pow(w, 1.0 / shape);
            }
        }

        public float GetGammaSingle(float shape, float scale)
        {
            // Implementation based on "A Simple Method for Generating Gamma Variables"
            // by George Marsaglia and Wai Wan Tsang.  ACM Transactions on Mathematical Software
            // Vol 26, No 3, September 2000, pages 363-372.

            float d, c, x, xsquared, v, u;

            if (shape >= 1f)
            {
                d = shape - 1f / 3f;
                c = 1f / MathF.Sqrt(9f * d);
                for (; ; )
                {
                    do
                    {
                        x = GetNormalSingle();
                        v = 1f + c * x;
                    }
                    while (v <= 0.0);
                    v = v * v * v;
                    u = GetUniformSingle();
                    xsquared = x * x;
                    if (u < 1.0 - .0331 * xsquared * xsquared || MathF.Log(u) < 0.5 * xsquared + d * (1.0 - v + MathF.Log(v)))
                        return scale * d * v;
                }
            }
            else if (shape <= 0f)
            {
                string msg = string.Format("Shape must be positive. Received {0}.", shape);
                throw new ArgumentOutOfRangeException(msg);
            }
            else
            {
                float g = GetGammaSingle(shape + 1f, 1f);
                float w = GetUniformSingle();
                return scale * g * MathF.Pow(w, 1f / shape);
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetChiSquare(double degreesOfFreedom)
        {
            // A chi squared distribution with n degrees of freedom
            // is a gamma distribution with shape n/2 and scale 2.
            return GetGamma(0.5 * degreesOfFreedom, 2.0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetChiSquareSingle(float degreesOfFreedom)
        {
            // A chi squared distribution with n degrees of freedom
            // is a gamma distribution with shape n/2 and scale 2.
            return GetGammaSingle(0.5f * degreesOfFreedom, 2f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetInverseGamma(double shape, double scale)
        {
            // If X is gamma(shape, scale) then
            // 1/Y is inverse gamma(shape, 1/scale)
            return 1.0 / GetGamma(shape, 1.0 / scale);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetInverseGamma(float shape, float scale)
        {
            // If X is gamma(shape, scale) then
            // 1/Y is inverse gamma(shape, 1/scale)
            return 1f / GetGammaSingle(shape, 1f / scale);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetWeibull(double shape, double scale)
        {
            if (shape <= 0.0 || scale <= 0.0)
            {
                string msg = string.Format("Shape and scale parameters must be positive. Recieved shape {0} and scale{1}.", shape, scale);
                throw new ArgumentOutOfRangeException(msg);
            }
            return scale * Math.Pow(-Math.Log(GetUniform()), 1.0 / shape);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetWeibullSingle(float shape, float scale)
        {
            if (shape <= 0f || scale <= 0f)
            {
                string msg = string.Format("Shape and scale parameters must be positive. Recieved shape {0} and scale{1}.", shape, scale);
                throw new ArgumentOutOfRangeException(msg);
            }
            return scale * MathF.Pow(-MathF.Log(GetUniformSingle()), 1f / shape);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetCauchy(double median, double scale)
        {
            if (scale <= 0)
            {
                string msg = string.Format("Scale must be positive. Received {0}.", scale);
                throw new ArgumentException(msg);
            }

            double p = GetUniform();

            // Apply inverse of the Cauchy distribution function to a uniform
            return median + scale * Math.Tan(Math.PI * (p - 0.5));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetCauchy(float median, float scale)
        {
            if (scale <= 0f)
            {
                string msg = string.Format("Scale must be positive. Received {0}.", scale);
                throw new ArgumentException(msg);
            }

            float p = GetUniformSingle();

            // Apply inverse of the Cauchy distribution function to a uniform
            return median + scale * MathF.Tan(MathF.PI * (p - 0.5f));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetStudentT(double degreesOfFreedom)
        {
            if (degreesOfFreedom <= 0)
            {
                string msg = string.Format("Degrees of freedom must be positive. Received {0}.", degreesOfFreedom);
                throw new ArgumentException(msg);
            }

            // See Seminumerical Algorithms by Knuth
            double y1 = GetNormal();
            double y2 = GetChiSquare(degreesOfFreedom);
            return y1 / Math.Sqrt(y2 / degreesOfFreedom);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetStudentT(float degreesOfFreedom)
        {
            if (degreesOfFreedom <= 0)
            {
                string msg = string.Format("Degrees of freedom must be positive. Received {0}.", degreesOfFreedom);
                throw new ArgumentException(msg);
            }

            // See Seminumerical Algorithms by Knuth
            float y1 = GetNormalSingle();
            float y2 = GetChiSquareSingle(degreesOfFreedom);
            return y1 / MathF.Sqrt(y2 / degreesOfFreedom);
        }

        // The Laplace distribution is also known as the double exponential distribution.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetLaplace(double mean, double scale)
        {
            double u = GetUniform();
            return (u < 0.5) ?
                mean + scale * Math.Log(2.0 * u) :
                mean - scale * Math.Log(2 * (1 - u));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetLaplaceSingle(float mean, float scale)
        {
            float u = GetUniformSingle();
            return (u < 0.5f) ?
                mean + scale * MathF.Log(2f * u) :
                mean - scale * MathF.Log(2f * (1f - u));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetLogNormal(double mu, double sigma)
        {
            return Math.Exp(GetNormal(mu, sigma));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetLogNormalSingle(float mu, float sigma)
        {
            return MathF.Exp(GetNormalSingle(mu, sigma));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetBeta(double a, double b)
        {
            if (a <= 0.0 || b <= 0.0)
            {
                string msg = string.Format("Beta parameters must be positive. Received {0} and {1}.", a, b);
                throw new ArgumentOutOfRangeException(msg);
            }

            // There are more efficient methods for generating beta samples.
            // However such methods are a little more efficient and much more complicated.
            // For an explanation of why the following method works, see
            // http://www.johndcook.com/distribution_chart.html#gamma_beta

            double u = GetGamma(a, 1.0);
            double v = GetGamma(b, 1.0);
            return u / (u + v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetBetaSingle(float a, float b)
        {
            if (a <= 0f || b <= 0f)
            {
                string msg = string.Format("Beta parameters must be positive. Received {0} and {1}.", a, b);
                throw new ArgumentOutOfRangeException(msg);
            }

            // There are more efficient methods for generating beta samples.
            // However such methods are a little more efficient and much more complicated.
            // For an explanation of why the following method works, see
            // http://www.johndcook.com/distribution_chart.html#gamma_beta

            float u = GetGammaSingle(a, 1f);
            float v = GetGammaSingle(b, 1f);
            return u / (u + v);
        }
    }
}
