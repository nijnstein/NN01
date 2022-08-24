using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;

namespace NN01
{
    static public partial class MathEx
    {
        public static float Max(this ReadOnlySpan<float> a)
        {
            float reff = MemoryMarshal.GetReference(a);
            return Max(MemoryMarshal.CreateSpan<float>(ref reff, a.Length)); 
        }

        public static float Max(this Span<float> a)
        {
            int i = 0;
            float result = float.MinValue;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Vector256<float> max = Vector256.Create(float.MinValue);
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~31))
                        {
                            max = Avx.Max(aa[j + 0], max);
                            max = Avx.Max(aa[j + 1], max);
                            max = Avx.Max(aa[j + 2], max);
                            max = Avx.Max(aa[j + 3], max);
                            i += 32;
                            j += 4;
                        }
                        while (i < (a.Length & ~7))
                        {
                            max = Avx.Max(aa[j], max);
                            i += 8;
                            j++;
                        }
                    }
                    result = HorizontalMaximum(max);
                }
            }
            unchecked
            {
                if (i < (a.Length & ~3))
                {
                    Span<float> results = stackalloc float[4];
                    while (i < (a.Length & ~3))
                    {
                        results[0] = a[i + 0] > results[0] ? a[i + 0] : results[0];
                        results[1] = a[i + 1] > results[1] ? a[i + 1] : results[0];
                        results[2] = a[i + 2] > results[2] ? a[i + 2] : results[0];
                        results[3] = a[i + 3] > results[3] ? a[i + 3] : results[0];
                        i += 4;
                    }
                    if (Sse.IsSupported)
                    {
                        result = HorizontalMaximum(MemoryMarshal.Cast<float, Vector128<float>>(results)[0]);
                    }
                    else
                    {
                        result = MathF.Max(results[0], MathF.Max(results[1], MathF.Max(results[2], results[3])));
                    }
                }
                while (i < a.Length)
                {
                    result = a[i] > result ? a[i] : result;
                    i++;
                }
            }
            return result;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HorizontalMaximum(this Vector128<float> v)
        {
            if (Sse.IsSupported)
            {
                // The control byte shuffles the four 32-bit floats of partialMax: ABCD -> BADC.
                Vector128<float> v1 = Sse.Shuffle(v, v, 0xB1);

                // Performs element-wise maximum operation: The 1st and 3rd 32-bit slots become
                // max(A, B) and max(C, D).
                Vector128<float> v2 = Sse.Max(v, v1);

                // The control byte shuffles the four 32-bit floats of v2: ABCD -> CAAA.
                v1 = Sse.Shuffle(v2, v2, 0x02);

                // Performs element-wise maximum operation: The 1st 32-bit slot becomes
                // max(A, B, C, D).
                return Sse.MaxScalar(v2, v1).GetElement(0);
            }
            else
            {
                return MathF.Max(v.GetElement(0), MathF.Max(v.GetElement(1), MathF.Max(v.GetElement(2), v.GetElement(3)))); 
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HorizontalMaximum(this Vector256<float> v)
        {
            if (Avx.IsSupported)
            {
                //                                                            /*      [upper   |   lower]                                                                         */ \
                //                                                            /*      [7 6 5 4 | 3 2 1 0]                                                                         */ \
                //    __m256 v1 = ymmA;                                       /* v1 = [H G F E | D C B A]                                                                         */ \
                //    __m256 v2 = _mm256_permute_ps(v1, 0b10'11'00'01);       /* v2 = [G H E F | C D A B]                                                                         */ \
                //    __m256 v3 = _mm256_max_ps(v1, v2);                      /* v3 = [W=max(G,H) W=max(G,H) Z=max(E,F) Z=max(E,F) | Y=max(C,D) Y=max(C,D) X=max(A,B) X=max(A,B)] */ \
                //                                                            /* v3 = [W W Z Z | Y Y X X]                                                                         */ \
                //    __m256 v4 = _mm256_permute_ps(v3, 0b00'00'10'10);       /* v4 = [Z Z W W | X X Y Y]                                                                         */ \
                //    __m256 v5 = _mm256_max_ps(v3, v4);                      /* v5 = [J=max(Z,W) J=max(Z,W) J=max(Z,W) J=max(Z,W) | I=max(X,Y) I=max(X,Y) I=max(X,Y) I=max(X,Y)] */ \
                //                                                            /* v5 = [J J J J | I I I I]                                                                         */ \
                //    __m128 v6 = _mm256_extractf128_ps(v5, 1);               /* v6 = [- - - - | J J J J]                                                                         */ \
                //    __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6); /* v7 = [- - - - | M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J)]                                     */ \
                //                                                            /* v7 = [- - - - | M M M M]                                                                         */ \
                //                                                            /* M = max(I,J)                                                                                     */ \
                //                                                            /* M = max(max(X,Y),max(Z,W))                                                                       */ \
                //                                                            /* M = max(max(max(A,B),max(C,D)),max(max(E,F),max(G,H)))                                           */ \
                //    _mm_store_ss(&result, v7);
                Vector256<float> v2 = Avx.Permute(v, 0b10110001);
                Vector256<float> v3 = Avx.Max(v, v2);
                Vector256<float> v4 = Avx.Permute(v3, 0b00001010);
                Vector256<float> v5 = Avx.Max(v3, v4);
                Vector128<float> v6 = Avx.Max(v5.GetLower(), v5.GetUpper());
                return v6.ToScalar();
            }
            else
            {
                return Math.Max(v.GetLower().HorizontalMaximum(), v.GetUpper().HorizontalMaximum()); 
            }
        }


        public static float Sum(this ReadOnlySpan<float> a)
        {
            float reff = MemoryMarshal.GetReference(a);
            return Sum(MemoryMarshal.CreateSpan<float>(ref reff, a.Length));
        }

        public static float Sum(this Span<float> a)
        {
            float sum = 0;
            unchecked
            {
                int i = 0;
                Span<float> sums = stackalloc float[4]; 

                while (i < (a.Length & ~3))
                {
                    sums[0] = a[i];
                    sums[1] = a[i + 1];
                    sums[2] = a[i + 2];
                    sums[3] = a[i + 3];
                    i += 4;
                }
               
                sum = sums[0] + sums[1] + sums[2] + sums[3]; 
                while (i < a.Length)
                {
                    sum += a[i]; 
                    i++;
                }
            }
            return sum;
        }

 
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum(Vector256<float> a)
        {
            if (Avx.IsSupported && Sse.IsSupported)
            {
                Vector256<float> v1 = Avx.HorizontalAdd(a, a);
                Vector256<float> v2 = Avx.HorizontalAdd(v1, v1);
                Vector128<float> v3 = Sse.AddScalar(v2.GetLower(), v2.GetUpper());
                return v3.GetElement(0);
            }
            else
            {
                return a.GetElement(0) + a.GetElement(1) + a.GetElement(2) + a.GetElement(3) +
                    a.GetElement(4) + a.GetElement(5) + a.GetElement(6) + a.GetElement(7); 
            }
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
