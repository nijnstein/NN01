using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;


using Avx2 = System.Runtime.Intrinsics.X86.Avx2;

namespace NN01 
{
    static public partial class MathEx
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(this float f)
        {
            f = MathF.Exp(f);
            return f / (1.0f + f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SigmoidDerivative(this float f)
        {
            return f * (1f - f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Swish(this float f)
        {
            return f * f.Sigmoid();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SwishDerivative(this float f)
        {
            // see
            // https://sefiks.com/2018/08/21/swish-as-neural-networks-activation-function/
            float e = 1f + MathF.Exp(-f);
            return (f * e + (e - f)) / (e * e);
        }


        public static float Max(this ReadOnlySpan<float> a)
        {
            float max = float.MinValue;
            unchecked
            {
                int i = 0;
                while (i < (a.Length & ~3))
                {
                    max = a[i] > max ? a[i] : max;
                    max = a[i + 1] > max ? a[i + 1] : max;
                    max = a[i + 2] > max ? a[i + 2] : max;
                    max = a[i + 3] > max ? a[i + 3] : max;
                    i += 4;
                }
                while (i < a.Length)
                {
                    max = a[i] > max ? a[i] : max;
                    i++;
                }
            }
            return max;
        }


        public static void Softmax(this Span<float> input, Span<float> output, bool stable = false)
        {
            // softmax == each exponent of every component divided by the sum of of all exponents 
            if (stable)
            {
                // stable done in other function,
                // - dont want inner loop branches
                // - dont want to modify input 
                // - dont want unneccessary memory operations on ALL non stable stofmax calls
                SoftmaxStable(input, output);
            }
            else
            {
                float summed = 0;
                for (int i = 0; i < input.Length; i++)
                {
                    float ex = MathF.Exp(input[i]);
                    summed += ex;
                    output[i] = ex;
                }
                for (int i = 0; i < input.Length; i++)
                {
                    output[i] = output[i] / summed;
                }
            }
        }

        /// <summary>
        /// stabelized softmax, max(input) is used to pull data into the negative in a reduced spread allowing exponentiation towards zero instead of infinity on large input values
        /// </summary>
        public static void SoftmaxStable(ReadOnlySpan<float> input, Span<float> output)
        {
            float summed = 0;
            float max = input.Max();

            for (int i = 0; i < input.Length; i++)
            {
                float ex = MathF.Exp(input[i]) - max;
                summed += ex;
                output[i] = ex;
            }
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = output[i] / summed;
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CDFNormal(this float x, float mean, float sd)
        {
            float p = 1f / MathF.Sqrt(2f * MathF.PI * (sd * sd));
            return p * MathF.Exp(-0.5f * ((x - mean) * (x - mean)) / (sd * sd));
        }



        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Normal(this Random random, float mean, float sd) => random.NextSingle().CDFNormal(mean, sd);


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Gaussian(this Random random, float mean, float sd)
        {
            // The method requires sampling from a uniform random of (0,1]
            // but Random.NextDouble() returns a sample of [0,1).
            float x1 = 1 - random.NextSingle();
            float x2 = 1 - random.NextSingle();

            // by replacing cos with sin we can generate a second distribution 
            float y1 = MathF.Sqrt(-2f * MathF.Log(x1)) * MathF.Cos(2f * (float)Math.PI * x2);
            return y1 * sd + mean;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Uniform(Span<float> a, float low, float high)
        {
            float d = (high - low - 1) / (a.Length - 1);
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = low + d * i;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Uniform(Span<float> a, float distribute)
        {
            float d = distribute / a.Length;
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = d;
            }
        }

        /// <summary>
        /// same as minmax over region 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HardTanH(this float f, float min = -1f, float max = 1f)
        {
            if (f < min) return min;
            if (f > max) return max;
            return f; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float MinMax(this float f, float min = -1f, float max = 1f)
        {
            return Math.Min(max, Math.Max(min, f)); 
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Range(this Random random, float low, float high) => random.NextSingle() * (high - low) + low;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Zero(this float[] a)
        {
            for (int i = 0; i < a.Length; i++) a[i] = 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Ones(this float[] a)
        {
            for (int i = 0; i < a.Length; i++) a[i] = 1f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp01(this float f)
        {
            return Math.Max(0, Math.Min(1, f));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp(this float f, float a, float b)
        {
            return Math.Max(a, Math.Min(b, f));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sign(this float f)
        {
            return f >= 0 ? 1 : -1;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMax(this Span<float> f)
        {
            float max = float.MinValue;
            int j = -1;
            unchecked
            {
                for (int i = 0; i < f.Length; i++)
                {
                    if (f[i] > max)
                    {
                        max = f[i];
                        j = i;
                    }
                }
            }
            return j;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMin(this Span<float> f)
        {
            float min = float.MaxValue;
            int j = -1; 
            unchecked
            {
                for (int i = 0; i < f.Length; i++)
                {
                    if (f[i] < min)
                    {
                        min = f[i];
                        j = i;
                    }
                }
            }
            return j;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMax(this ReadOnlySpan<float> f)
        {
            float max = float.MinValue;
            int j = -1;
            unchecked
            {
                for (int i = 0; i < f.Length; i++)
                {
                    if (f[i] > max)
                    {
                        max = f[i];
                        j = i;
                    }
                }
            }
            return j;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMin(this ReadOnlySpan<float> f)
        {
            float min = float.MaxValue;
            int j = -1;
            unchecked
            {
                for (int i = 0; i < f.Length; i++)
                {
                    if (f[i] < min)
                    {
                        min = f[i];
                        j = i;
                    }
                }
            }
            return j;
        }
    }
}
