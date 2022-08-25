using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;

namespace NN01
{
    static public partial class Intrinsics
    {
        public static float Sum(ReadOnlySpan<float> a)
        {
            float reff = MemoryMarshal.GetReference(a);
            return Sum(MemoryMarshal.CreateSpan<float>(ref reff, a.Length));
        }

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
    }
}
