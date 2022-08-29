﻿using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using System.Diagnostics.Contracts;
using static System.Formats.Asn1.AsnWriter;
using System.Diagnostics;
using System.Xml.Linq;

namespace NN01
{
    static public partial class MathEx
    {                                                                                 

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Max(Span<float> a)
        {
            return Intrinsics.MaxUnaligned(a);
            // assume unaligned if called through mathex


            // in unsafe builds we could check for alignment on 256 borders then use intrinsics
            // 
            //  so  if using   MathEx.Max()  then if aligned  MathEx.Max uses Intrinsics 
            // 
            // return Intrinsics.Max(a); 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(Span<float> a)
        {
            return Intrinsics.SumUnaligned(a);
        }

        /// <summary>
        /// only valid between -80 and 80
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> ExpFast(Span<float> a)
        {
            return Intrinsics.Exp(a, a);
        }

        /// <summary>
        /// only valid between -80 and 80
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> ExpFast(Span<float> a, Span<float> output)
        {
            return Intrinsics.Exp(a, output);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Exp(Span<float> a, Span<float> output)
        {
            unchecked
            {
                for (int i = 0; i < a.Length; i++)
                {
                    output[i] = MathF.Exp(a[i]);
                }
            }
            return output;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Exp(Span<float> a)
        {
            return Exp(a, a);
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
        public static void SoftmaxStable(Span<float> input, Span<float> output)
        {
            float summed = 0;
            float max = Max(input); 

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
        public static int ArgMax(this float[] f)
        {
            return ArgMax(f.AsSpan()); 
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
        public static int ArgMin(this float[] f)
        {
            return ArgMin(f.AsSpan());
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

        [Pure][MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Square(this float f) => f * f;

        [Pure][MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Square(this double f) => f * f;

        [Pure][MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Square(this int f) => f * f;

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Average(this float[][] input)
        {
            int c = 0;
            float average = 0;
            unchecked
            {
                for (int i = 0; i < input.Length; i++)
                {
                    average += Intrinsics.Sum(input[i]);
                    c += input[i].Length;
                }
            }
            return average / c;
        }



        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Average(this Span<float> a) => Intrinsics.Sum(a) / a.Length; 
        
        [Pure]                                            
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance(this Span<float> a) => Variance(a, Average(a));

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance(this float[][] a, float average)
        {
            int c = 0; 
            float sq = 0; 
            unchecked
            {
                for(int i = 0; i < a.Length; i++)
                {
                    sq += Intrinsics.SumSquaredDifferences(a[i], average);
                    c += a.Length; 
                }
            }
            return sq / c;
        }

        /// <summary>
        /// V =  sum((a - mean)^2) / a.length
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance(this Span<float> a, float average) => Intrinsics.SumSquaredDifferences(a, average) / a.Length;

        /// <summary>
        /// V =  sum((a - mean)^2)  /  sum(a)
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float VarianceN(this Span<float> a, float average) => Intrinsics.SumSquaredDifferences(a, average) / Intrinsics.Sum(a);

        /// <summary>
        /// assumes all elements of a sum to a total of 1, variance is then just the summed squared differences
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance1(this Span<float> a, float average) => Intrinsics.SumSquaredDifferences(a, average);

        /// <summary>
        /// euclidian distance between 2 vectors 
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Euclid(this Span<float> a, Span<float> b)
        {
            Debug.Assert(a.Length == b.Length);
            float f = 0f; 
            unchecked
            {
                for (int i = 0; i < a.Length; i++)
                {
                    f += MathF.Pow(MathF.Abs(a[i] - b[i]), 2);
                }
            }
            return MathF.Sqrt(f);
        }


        /// <summary>
        /// euclidian distance between 2 vectors 
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Euclid(this Span<byte> a, Span<byte> b)
        {
            Debug.Assert(a.Length == b.Length); 
            float f = 0f;
            unchecked
            {
                for (int i = 0; i < a.Length; i++)
                {
                    f += MathF.Pow(MathF.Abs(a[i] - b[i]), 2);
                }
            }
            return MathF.Sqrt(f);
        }

        /// <summary>
        /// Chebyshev distance between 2 vectors 
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Chebyshev(this byte[] a, byte[] b)
        {
            Debug.Assert(a.Length == b.Length);
            int max = int.MinValue;
            unchecked
            {
                for (int i = 0; i < a.Length; i++)
                {
                    int d = Math.Abs(a[i] - b[i]);
                    if (d > max) max = d;
                }
            }
            return max;
        }

        /// <summary>
        /// Chebyshev distance between 2 vectors 
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Chebyshev(this Span<float> a, Span<float> b) 
        {
            Debug.Assert(a.Length == b.Length);
            float max = float.MinValue;
            unchecked
            {
                for (int i = 0; i < a.Length; i++)
                {
                    float d = MathF.Abs(a[i] - b[i]);
                    if (d > max) max = d;
                }
            }
            return max;
        }
        
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Manhattan(this Span<float> a, Span<float> b)
        {
            Debug.Assert(a.Length == b.Length);
            float f = 0f;
            unchecked
            {
                for (int i = 0; i < a.Length; i++)
                {
                    f += MathF.Abs(a[i] - b[i]);
                }
            }
            return f;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Manhattan(this Span<byte> a, Span<byte> b)
        {
            Debug.Assert(a.Length == b.Length);
            float f = 0f;
            unchecked
            {
                for (int i = 0; i < a.Length; i++)
                {
                    f += MathF.Abs(a[i] - b[i]);
                }
            }
            return f;
        }


        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Rmse(Span<float> truth, Span<float> prediction)
        {
            Debug.Assert(truth.Length != prediction.Length, $"length doesn't match, {truth.Length} != {prediction.Length}");
            Debug.Assert(truth.Length > 0);

            float rmse = MathF.Sqrt(Intrinsics.AverageSquaredDifferences(truth, prediction)); 
            return rmse;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mape(Span<float> truth, Span<float> prediction)
        {
            Debug.Assert(truth.Length != prediction.Length, $"length doesn't match, {truth.Length} != {prediction.Length}");
            Debug.Assert(truth.Length > 0);

            int i = 0;
            float f = 0; 

            while(i < truth.Length)
            {
                f += Math.Abs(truth[i] - prediction[i]) / truth[i];
                i++; 
            }

            return f / truth.Length;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mae(float[] truth, float[] prediction)
        {
            Debug.Assert(truth.Length != prediction.Length, $"length doesn't match, {truth.Length} != {prediction.Length}");
            Debug.Assert(truth.Length > 0); 

            return Intrinsics.SumAbsoluteDifferences(truth, prediction) / truth.Length;
        }


        /// <summary>
        /// interpolate between a 2d maps of different sizes
        /// w * h must be equal to input length
        /// </summary>
        public static float[] BiLinearInterpolate(this float[] input, int w, int h, int outw, int outh)
        {
            Debug.Assert(input.Length == w * h);
            return input.BiLinearInterpolate(0, 0, w, w, h, outw, outh);
        }

        /// <summary>
        /// interpolate between a 2d region in on array to another 2d region of different dimensions
        /// </summary>
        public static float[] BiLinearInterpolate(this float[] input, int x0, int y0, int xdim, int w, int h, int outw, int outh)
        {
            return BiLinearInterpolate(input, new float[outw * outh], x0, y0, xdim, w, h, outw, outh);
        }

        /// <summary>
        /// interpolate between a 2d region in on array to another 2d region of different dimensions
        /// </summary>
        public static float[] BiLinearInterpolate(this float[] input, float[] output, int x0, int y0, int xdim, int w, int h, int outw, int outh)
        {
            float xr = (float)(w - 1f) / outw;
            float yr = (float)(h - 1f) / outh;

            unchecked
            {
                for (int iy = 0; iy < outh; iy++)
                {
                    for (int ix = 0; ix < outw; ix++)
                    {
                        int x = (int)(xr * ix);
                        int y = (int)(yr * iy);
                        float x_diff = (xr * ix) - x;
                        float y_diff = (yr * iy) - y;
                        int idx = (y + y0) * xdim + x + x0;

                        output[iy * outw + ix] =
                            input[idx] * (1 - x_diff) * (1 - y_diff) +
                            input[idx + 1] * x_diff * (1 - y_diff) +
                            input[idx + xdim] * y_diff * (1 - x_diff) +
                            input[idx + xdim + 1] * x_diff * y_diff;
                    }
                }
            }
            return output;
        }


        /// <summary>
        /// in-place 
        /// </summary>
        /// <returns>same array object</returns>
        public static double[] Shift2D(this double[] input, int width, int height)
        {
            for (int y = 0; y < height; y++)
            {
                int y2 = (y + height / 2) % height;
                for (int x = 0; x < width; x++)
                {
                    int x2 = (x + width / 2) % width;
                    input[x + y * width] = input[x2 + y2 * width];
                }
            }
            return input;
        }

        /// <summary>
        /// align to a 256 bit boundary 
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static public int Align256(this int i) => ((i + 31) / 32) * 32;


        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static public float ExpFast256(float x)
        {
            x = 1f + x / 256f;
            x *= x; x *= x; x *= x; x *= x;
            x *= x; x *= x; x *= x; x *= x;
            return x;
        }
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static public float ExpFast1024(float x)
        {
            x = 1f + x / 1024f;
            x *= x; x *= x; x *= x; x *= x;
            x *= x; x *= x; x *= x; x *= x;
            x *= x; x *= x;
            return x;
        }

    }
}


  