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
        /// a = a * b
        /// </summary>
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span<float> MultiplyScalar(Span<float> a, float b)
        {
            return MultiplyScalar(a, b, a);
        }

        /// <summary>
        /// output = a * b
        /// </summary>
        public static Span<float> MultiplyScalar(Span<float> a, float b, Span<float> output)
        {
            int i = 0; 

            if(Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Vector256<float> bb = Vector256.Create(b); 
                    Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(output);
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~31))
                        {
                            d[j + 0] = Avx.Multiply(aa[j + 0], bb);
                            d[j + 1] = Avx.Multiply(aa[j + 1], bb);
                            d[j + 2] = Avx.Multiply(aa[j + 2], bb);
                            d[j + 3] = Avx.Multiply(aa[j + 3], bb);
                            i += 32;
                            j += 4;
                        }
                        while (i < (a.Length & ~7))
                        {
                            d[j] = Avx.Multiply(aa[j], bb);
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
                    output[i + 0] = a[i + 0] * b;
                    output[i + 1] = a[i + 1] * b;
                    output[i + 2] = a[i + 2] * b;
                    output[i + 3] = a[i + 3] * b;
                    i += 4;
                }
                while (i < a.Length)
                {
                    output[i] = a[i] * b;
                    i++;
                }
            }

            return output; 
        }
    }
}
