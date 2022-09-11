using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using System.Diagnostics.Contracts;

namespace NSS
{
    static public partial class Intrinsics
    {
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(ReadOnlySpan<float> a)
        {
            float reff = MemoryMarshal.GetReference(a);
            return Sum(MemoryMarshal.CreateSpan<float>(ref reff, a.Length));
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SumUnaligned(Span<float> a)
        {
            float sum = 0;
            unchecked
            {
                int i = 0;
                Span<float> sums = stackalloc float[4];
                sums[0] = 0;
                sums[1] = 0;
                sums[2] = 0;
                sums[3] = 0;

                while (i < (a.Length & ~3))
                {
                    sums[0] += a[i];
                    sums[1] += a[i + 1];
                    sums[2] += a[i + 2];
                    sums[3] += a[i + 3];
                    i += 4;
                }
               
                sum += sums[0] + sums[1] + sums[2] + sums[3]; 
                while (i < a.Length)
                {
                    sum += a[i]; 
                    i++;
                }
            }
            return sum;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(Span<float> a)
        {
            float sum = 0;
            int i = 0; 

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Vector256<float> b = Vector256<float>.Zero;
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~31))
                        {
                            b = Avx.Add(aa[j + 0], 
                                    Avx.Add(aa[j + 1], 
                                        Avx.Add(aa[j + 2], 
                                            Avx.Add(aa[j + 3], b))));
                            j += 4;
                            i += 32;
                        }
                        while (i < (a.Length & ~7))
                        {
                            b = Avx.Add(b, aa[j]);
                            j++;
                            i += 8; 
                        }
                        sum = HorizontalSum(b);
                    }
                }
            }

            unchecked
            {
                Span<float> sums = stackalloc float[4];
                sums[0] = 0;
                sums[1] = 0;
                sums[2] = 0;
                sums[3] = 0;

                while (i < (a.Length & ~3))
                {
                    sums[0] += a[i];
                    sums[1] += a[i + 1];
                    sums[2] += a[i + 2];
                    sums[3] += a[i + 3];
                    i += 4;
                }
                sum += sums[0] + sums[1] + sums[2] + sums[3];
                while (i < a.Length)
                {
                    sum += a[i];
                    i++;
                }
            }
            return sum;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HorizontalSum(Vector256<float> a)
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

        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> Sum(in Vector256<float> v)
        {
            Vector256<float> p = Avx.HorizontalAdd(v, v);
            // p = 1000.1000

            return Avx.HorizontalAdd(p, p);
            // ret (1000.0000
        }

        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        internal static Vector128<float> Sum(in Vector128<float> v)
        {
            if (Sse3.IsSupported)
            {
                Vector128<float> p = Sse3.HorizontalAdd(v, v);
                return Sse3.HorizontalAdd(p, p);
            }
            else
            {
                Vector128<float> p = Sse.Add(v, Sse.MoveHighToLow(v, v));
                // ABCD -> BADC.
                return Sse.Add(p, Sse.Shuffle(p, p, 0xB1));
            }
        }

        [Pure] 
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> VectorMax256(in Vector256<float> vector)
        {
            // The control byte shuffles the eight 32-bit floats of partialMax: ABCD|EFGH -> BADC|FEHG.
            Vector256<float> x1 = Avx.Shuffle(vector, vector, 0xB1);

            // Performs element-wise maximum operation: The 1st, 3rd, 5th, and 7th 32-bit slots become
            // max(A, B), max(C, D), max(E, F), and max(G, H).
            Vector256<float> partialMax = Avx.Max(vector, x1);

            // The control byte shuffles the eight 32-bit floats of partialMax: ABCD|EFGH -> CAAA|GEEE.
            x1 = Avx.Shuffle(partialMax, partialMax, 0x02);

            // Performs element-wise maximum operation: The 1st and 5th 32-bit slots become
            // max(max(A, B), max(C, D)) = max(A, B, C, D) and
            // max(max(E, F), max(G, H)) = max(E, F, G, H).
            return Avx.Max(partialMax, x1);
        }



        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static float SumSquaredDifferences(Span<float> a, float mean)
        {
            float sum = 0;
            int i = 0;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Vector256<float> mean256 = Vector256.Create(mean); 
                    Vector256<float> b = Vector256<float>.Zero;
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~7))
                        {
                            Vector256<float> v = Avx.Subtract(aa[j], mean256);
                            b = MultiplyAdd(v, v, b);

                            j++;
                            i += 8;
                        }
                        sum = HorizontalSum(b);
                    }
                }
            }

            unchecked
            {
                Span<float> sums = stackalloc float[4];
                sums[0] = 0;
                sums[1] = 0;
                sums[2] = 0;
                sums[3] = 0;

                while (i < (a.Length & ~3))
                {
                    sums[0] += (a[i + 0] - mean).Square();
                    sums[1] += (a[i + 1] - mean).Square();
                    sums[2] += (a[i + 2] - mean).Square();
                    sums[3] += (a[i + 3] - mean).Square();
                    i += 4;
                }
                sum += sums[0] + sums[1] + sums[2] + sums[3];
                while (i < a.Length)
                {
                    sum += (a[i] - mean).Square();
                    i++;
                }
            }
            return sum;
        }

        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static float SumSquaredDifferences(Span<float> a, Span<float> b)
        {
            float sum = 0;
            int i = 0;
            int j = 0;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> bb = MemoryMarshal.Cast<float, Vector256<float>>(b);
                    Vector256<float> c = Vector256<float>.Zero;
                    unchecked
                    {
                        while (i < (a.Length & ~7))
                        {
                            Vector256<float> v256 = Avx.Subtract(aa[j], bb[j]);
                            c = MultiplyAdd(v256, v256, c);
                            j++;
                            i += 8;
                        }
                        sum = HorizontalSum(c);
                    }
                }
            }          

            unchecked
            {
                Span<float> sums = stackalloc float[4];
                sums[0] = 0;
                sums[1] = 0;
                sums[2] = 0;
                sums[3] = 0;

                while (i < (a.Length & ~3))
                {
                    sums[0] += (a[i + 0] - b[i + 0]).Square();
                    sums[1] += (a[i + 1] - b[i + 1]).Square();
                    sums[2] += (a[i + 2] - b[i + 2]).Square();
                    sums[3] += (a[i + 3] - b[i + 3]).Square();
                    i += 4;
                }
                sum += sums[0] + sums[1] + sums[2] + sums[3];
                while (i < a.Length)
                {
                    sum += (a[i] - b[i]).Square();
                    i++;
                }
            }
            return sum;
        }

        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static float SumAbsoluteDifferences(Span<float> a, Span<float> b)
        {
            float sum = 0;
            int i = 0;
            int j = 0;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> bb = MemoryMarshal.Cast<float, Vector256<float>>(b);
                    Vector256<float> c = Vector256<float>.Zero;
                    Vector256<float> sign = Vector256.Create(-0f); 
                    unchecked
                    {
                        while (i < (a.Length & ~7))
                        {
                            c = Avx.Add(Avx.And(Avx.Subtract(aa[j], bb[j]), sign), c);
                            j++;
                            i += 8;
                        }
                        sum = HorizontalSum(c);
                    }
                }
            }

            unchecked
            {
                Span<float> sums = stackalloc float[4];
                sums[0] = 0;
                sums[1] = 0;
                sums[2] = 0;
                sums[3] = 0;

                while (i < (a.Length & ~3))
                {
                    sums[0] += Math.Abs(a[i + 0] - b[i + 0]);
                    sums[1] += Math.Abs(a[i + 1] - b[i + 1]);
                    sums[2] += Math.Abs(a[i + 2] - b[i + 2]);
                    sums[3] += Math.Abs(a[i + 3] - b[i + 3]);
                    i += 4;
                }
                sum += sums[0] + sums[1] + sums[2] + sums[3];
                while (i < a.Length)
                {
                    sum += Math.Abs(a[i] - b[i]);
                    i++;
                }
            }
            return sum;
        }

        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static float SumSquares(Span<float> a)
        {
            float sum = 0;
            int i = 0;
            int j = 0;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Vector256<float> c = Vector256<float>.Zero;
                    unchecked
                    {
                        while (i < (a.Length & ~7))
                        {
                            c = MultiplyAdd(aa[j], aa[j], c);
                            j++;
                            i += 8;
                        }
                        sum = HorizontalSum(c);
                    }
                }
            }

            unchecked
            {
                Span<float> sums = stackalloc float[4];
                sums[0] = 0;
                sums[1] = 0;
                sums[2] = 0;
                sums[3] = 0;

                while (i < (a.Length & ~3))
                {
                    sums[0] += a[i + 0].Square();
                    sums[1] += a[i + 1].Square();
                    sums[2] += a[i + 2].Square();
                    sums[3] += a[i + 3].Square();
                    i += 4;
                }
                sum += sums[0] + sums[1] + sums[2] + sums[3];
                while (i < a.Length)
                {
                    sum += a[i].Square();
                    i++;
                }
            }
            return sum;
        }

        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static float SumWeighted(Span<float> a, Span<float> b)
        {
            float sum = 0;
            int i = 0;
            int j = 0;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> bb = MemoryMarshal.Cast<float, Vector256<float>>(b);
                    Vector256<float> c = Vector256<float>.Zero;
                    unchecked
                    {
                        while (i < (a.Length & ~7))
                        {
                            c = MultiplyAdd(aa[j], bb[j], c);
                            j++;
                            i += 8;
                        }
                        sum = HorizontalSum(c);
                    }
                }
            }
            unchecked
            {
                Span<float> sums = stackalloc float[4];
                sums[0] = 0;
                sums[1] = 0;
                sums[2] = 0;
                sums[3] = 0;

                while (i < (a.Length & ~3))
                {
                    sums[0] += a[i + 0] * b[i + 0];
                    sums[1] += a[i + 1] * b[i + 1];
                    sums[2] += a[i + 2] * b[i + 2];
                    sums[3] += a[i + 3] * b[i + 3];
                    i += 4;
                }
                sum += sums[0] + sums[1] + sums[2] + sums[3];
                while (i < a.Length)
                {
                    sum += a[i] * b[i];
                    i++;
                }
            }
            return sum;
        }
    }
}
