using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace NN01
{
    static public partial class Intrinsics
    {

        /// <summary>
        /// => make sure data is aligned 
        /// c = a * b + c
        /// </summary>
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span<float> MultiplyAdd(Span<float> a, Span<float> b, Span<float> c)
        {
            return Intrinsics.MultiplyAdd(a, b, c, c);
        }

        /// <summary>
        /// => make sure data is aligned 
        /// output = a * b + c
        /// </summary>
        public static Span<float> MultiplyAdd(Span<float> a, Span<float> b, Span<float> c, Span<float> output)
        {
            int i = 0; 

            if(Avx.IsSupported || Sse.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> bb = MemoryMarshal.Cast<float, Vector256<float>>(b);
                    Span<Vector256<float>> cc = MemoryMarshal.Cast<float, Vector256<float>>(c);
                    Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(output);
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~31))
                        {
                            d[j + 0] = MultiplyAdd(aa[j + 0], bb[j + 0], cc[j + 0]);
                            d[j + 1] = MultiplyAdd(aa[j + 1], bb[j + 1], cc[j + 1]);
                            d[j + 2] = MultiplyAdd(aa[j + 2], bb[j + 2], cc[j + 2]);
                            d[j + 3] = MultiplyAdd(aa[j + 3], bb[j + 3], cc[j + 3]);
                            i += 32;
                            j += 4;
                        }
                        while (i < (a.Length & ~7))
                        {
                            d[j] = MultiplyAdd(aa[j], bb[j], cc[j]);
                            i += 8;
                            j++;
                        }
                    }
                }
            }

            unchecked
            {
                while (i < (a.Length & ~3))
                {
                    output[i + 0] = MultiplyAdd(a[i + 0], b[i + 0], c[i + 0]);
                    output[i + 1] = MultiplyAdd(a[i + 1], b[i + 1], c[i + 1]);
                    output[i + 2] = MultiplyAdd(a[i + 2], b[i + 2], c[i + 2]);
                    output[i + 3] = MultiplyAdd(a[i + 3], b[i + 3], c[i + 3]);
                    i += 4;
                }
                while (i < a.Length)
                {
                    output[i] = MultiplyAdd(a[i], b[i], c[i]);
                    i++;
                }
            }

            return output; 
        }

        /// <summary>
        /// output = a * b + c
        /// </summary>
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static float MultiplyAdd(float a, float b, float c)
        {
            return a * b + c;
        }

        /// <summary>
        /// output = a * b + c
        /// </summary>
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
        {
            if (Fma.IsSupported)
            {
                return Fma.MultiplyAdd(a, b, c);
            }
            else
            {
                if (Avx.IsSupported)
                {
                    return Avx.Add(Avx.Multiply(a, b), c);
                }
                if(Sse.IsSupported)
                {
                    Vector128<float> low = Sse.Add(Sse.Multiply(a.GetLower(), b.GetLower()), c.GetLower());
                    Vector128<float> high = Sse.Add(Sse.Multiply(a.GetUpper(), b.GetUpper()), c.GetUpper());
                    return Vector256.Create(low, high); 
                }
                else
                {
                    Span<float> f = stackalloc float[8];
                    for (int i = 0; i < 8; i++)
                    {
                        f[i] = a.GetElement(i) * b.GetElement(i) + c.GetElement(i);  
                    }
                    return MemoryMarshal.Cast<float, Vector256<float>>(f)[0];
                }
            }
        }
    }
}
