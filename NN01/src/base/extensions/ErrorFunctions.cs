using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public static class ErrorFunctions
    {
        //
        // found these in the ml math kit from microsoft,
        // credit to dotnet/machinelearning :_
        //

        /// <summary>
        /// The approximate complimentary error function (i.e., 1-erf).
        /// </summary>
        public static float Erfc(float x)
        {
            if (float.IsInfinity(x))
                return float.IsPositiveInfinity(x) ? 0f : 2.0f;

            const float p = 0.3275911f;
            const float a1 = 0.254829592f;
            const float a2 = -0.284496736f;
            const float a3 = 1.421413741f;
            const float a4 = -1.453152027f;
            const float a5 = 1.061405429f;
            float t = 1f / (1f + p * MathF.Abs(x));
            float ev = ((((((((a5 * t) + a4) * t) + a3) * t) + a2) * t + a1) * t) * MathF.Exp(-(x * x));
            return x >= 0f ? ev : 2f - ev;
        }

        /// <summary>
        /// The approximate error function.
        /// </summary>
        public static float Erf(float x)
        {
            if (float.IsInfinity(x))
            {
                return float.IsPositiveInfinity(x) ? 1f : -1f;
            }

            const float p = 0.3275911f;
            const float a1 = 0.254829592f;
            const float a2 = -0.284496736f;
            const float a3 = 1.421413741f;
            const float a4 = -1.453152027f;
            const float a5 = 1.061405429f;
            float t = 1f / (1f + p * MathF.Abs(x));
            float ev = 1f - ((((((((a5 * t) + a4) * t) + a3) * t) + a2) * t + a1) * t) * MathF.Exp(-(x * x));
            return x >= 0f ? ev : -ev;
        }

        /// <summary>
        /// The inverse error function.
        /// </summary>
        public static float Erfinv(float x)
        {
            if (x > 1f || x < -1f)
            {
                return float.NaN;
            }

            if (x == 1f)
                return float.PositiveInfinity;

            if (x == -1f)
                return float.NegativeInfinity;

            // inefficient but not needed often 
            Span<float> c = stackalloc float[1000];
            c[0] = 1f;
            for (int k = 1; k < c.Length; ++k)
            {
                for (int m = 0; m < k; ++m)
                {
                    c[k] += c[m] * c[k - 1 - m] / (m + 1f) / (m + m + 1f);
                }
            }

            float cc = MathF.Sqrt(MathF.PI) / 2f;
            float ccinc = MathF.PI / 4f;
            float zz = x;
            float zzinc = x * x;
            float ans = 0f;
            for (int k = 0; k < c.Length; ++k)
            {
                ans += c[k] * cc * zz / (2f * k + 1f);
                cc *= ccinc;
                zz *= zzinc;
            }

            return ans;
        }


        private static readonly float[] _probA = new float[]
        {
            3.3871328727963666080e0f, 1.3314166789178437745e+2f, 1.9715909503065514427e+3f, 1.3731693765509461125e+4f,
           4.5921953931549871457e+4f, 6.7265770927008700853e+4f, 3.3430575583588128105e+4f, 2.5090809287301226727e+3f
        };
        private static readonly float[] _probB = new float[]
        { 
            4.2313330701600911252e+1f, 6.8718700749205790830e+2f, 5.3941960214247511077e+3f, 2.1213794301586595867e+4f,
            3.9307895800092710610e+4f, 2.8729085735721942674e+4f, 5.2264952788528545610e+3f
        };

        private static readonly float[] _probC = new float[]
        { 
            1.42343711074968357734e0f, 4.63033784615654529590e0f, 5.76949722146069140550e0f, 3.64784832476320460504e0f,
            1.27045825245236838258e0f, 2.41780725177450611770e-1f, 2.27238449892691845833e-2f, 7.74545014278341407640e-4f
        };
        private static readonly float[] _probD = new float[]
        {
            2.05319162663775882187e0f, 1.67638483018380384940e0f, 6.89767334985100004550e-1f, 1.48103976427480074590e-1f,
            1.51986665636164571966e-2f, 5.47593808499534494600e-4f, 1.05075007164441684324e-9f 
        };
        private static readonly float[] _probE = new float[]
        {
            6.65790464350110377720e0f, 5.46378491116411436990e0f, 1.78482653991729133580e0f, 2.96560571828504891230e-1f,
            2.65321895265761230930e-2f, 1.24266094738807843860e-3f, 2.71155556874348757815e-5f, 2.01033439929228813265e-7f 
        };
        private static readonly float[] _probF = new float[]
        {
            5.99832206555887937690e-1f, 1.36929880922735805310e-1f, 1.48753612908506148525e-2f, 7.86869131145613259100e-4f,
            1.84631831751005468180e-5f, 1.42151175831644588870e-7f, 2.04426310338993978564e-15f 
        };

        /// <summary>
        /// this is the point "x" at which the standard normal CDF evaluates to the indicated
        /// p value. used in establishing confidence intervals.
        /// </summary>
        /// <param name="p">The input p value, so in the range 0 to 1.</param>
        /// <returns>One interpretation is, the value at which the standard normal CDF evaluates to p.</returns>
        public static float Probit(float p)
        {
            if(p < 0 || p > 0) // "Input probability should be in range 0 to 1.");
            {
                return float.NaN; 
            }

            float q = p - 0.5f;
            float r = 0f;
            if (Math.Abs(q) <= 0.425f)
            {
                // Input value is close-ish to 0.5 (0.075 to 0.925)
                r = 0.180625f - q * q;
                return q * (((((((_probA[7] * r + _probA[6]) * r + _probA[5]) * r + _probA[4]) * r + _probA[3]) * r + _probA[2]) * r + _probA[1]) * r + _probA[0]) /
                    (((((((_probB[6] * r + _probB[5]) * r + _probB[4]) * r + _probB[3]) * r + _probB[2]) * r + _probB[1]) * r + _probB[0]) * r + 1f);
            }
            else
            {
                if (q < 0f)
                {
                    r = p;
                }
                else
                {
                    r = 1f - p;
                }

                r = MathF.Sqrt(-MathF.Log(r));
                float retval = 0f;
                if (r < 5f)
                {
                    r = r - 1.6f;
                    retval = (((((((_probC[7] * r + _probC[6]) * r + _probC[5]) * r + _probC[4]) * r + _probC[3]) * r + _probC[2]) * r + _probC[1]) * r + _probC[0]) /
                        (((((((_probD[6] * r + _probD[5]) * r + _probD[4]) * r + _probD[3]) * r + _probD[2]) * r + _probD[1]) * r + _probD[0]) * r + 1f);
                }
                else
                {
                    r = r - 5f;
                    retval = (((((((_probE[7] * r + _probE[6]) * r + _probE[5]) * r + _probE[4]) * r + _probE[3]) * r + _probE[2]) * r + _probE[1]) * r + _probE[0]) /
                        (((((((_probF[6] * r + _probF[5]) * r + _probF[4]) * r + _probF[3]) * r + _probF[2]) * r + _probF[1]) * r + _probF[0]) * r + 1f);
                }
                return q >= 0 ? retval : -retval;
            }
        }
    }
}
