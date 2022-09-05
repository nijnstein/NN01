using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;

namespace NSS
{
    static public partial class Intrinsics
    {

        public static float Max(ReadOnlySpan<float> a)
        {
            float reff = MemoryMarshal.GetReference(a);
            return Intrinsics.Max(MemoryMarshal.CreateSpan<float>(ref reff, a.Length)); 
        }

        public static float Max(Span<float> a)
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
                    result = Intrinsics.HorizontalMaximum(max);
                }
            }
            unchecked
            {
                if (i < (a.Length & ~3))
                {
                    Span<float> results = stackalloc float[4];
                    results[0] = float.MinValue;
                    results[1] = float.MinValue;
                    results[2] = float.MinValue;
                    results[3] = float.MinValue;
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
                        result = Intrinsics.HorizontalMaximum(MemoryMarshal.Cast<float, Vector128<float>>(results)[0]);
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
        public static float MaxUnaligned(Span<float> a)
        {
            int i = 0;
            float result = 0; 

            if (i < (a.Length & ~3))
            {
                Span<float> results = stackalloc float[4];
                results[0] = float.MinValue;
                results[1] = float.MinValue;
                results[2] = float.MinValue;
                results[3] = float.MinValue;
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
                    result = Intrinsics.HorizontalMaximum(MemoryMarshal.Cast<float, Vector128<float>>(results)[0]);
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
            return result; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HorizontalMaximum(Vector128<float> v)
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
        public static float HorizontalMaximum(Vector256<float> v)
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
                return Math.Max(HorizontalMaximum(v.GetLower()), HorizontalMaximum(v.GetUpper())); 
            }
        }


    }
}
