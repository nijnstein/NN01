﻿using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace NSS
{
    static public partial class Intrinsics
    {

        /// <summary>
        /// a = a / b
        /// </summary>
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span<float> DivideScalar(Span<float> a, float b)
        {
            return DivideScalar(a, b, a);
        }

        /// <summary>
        /// b = a / b
        /// </summary>
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span<float> DivideScalar(float a, Span<float> b)
        {
            return DivideScalar(a, b, b);
        }

        /// <summary>
        /// output = a / b
        /// </summary>
        public static Span<float> DivideScalar(Span<float> a, float b, Span<float> output)
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
                            d[j + 0] = Avx.Divide(aa[j + 0], bb);
                            d[j + 1] = Avx.Divide(aa[j + 1], bb);
                            d[j + 2] = Avx.Divide(aa[j + 2], bb);
                            d[j + 3] = Avx.Divide(aa[j + 3], bb);
                            i += 32;
                            j += 4;
                        }
                        while (i < (a.Length & ~7))
                        {
                            d[j] = Avx.Divide(aa[j], bb);
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
                    output[i + 0] = a[i + 0] / b;
                    output[i + 1] = a[i + 1] / b;
                    output[i + 2] = a[i + 2] / b;
                    output[i + 3] = a[i + 3] / b;
                    i += 4;
                }
                while (i < a.Length)
                {
                    output[i] = a[i] / b;
                    i++;
                }
            }

            return output; 
        }


        /// <summary>
        /// b = a / b
        /// </summary>
        public static Span<float> DivideScalar(float a, Span<float> b, Span<float> output)
        {
            int i = 0;

            if (Avx.IsSupported)
            {
                if (b.Length > 7)
                {
                    Span<Vector256<float>> bb = MemoryMarshal.Cast<float, Vector256<float>>(b);
                    Vector256<float> aa = Vector256.Create(a);
                    Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(output);
                    unchecked
                    {
                        int j = 0;
                        while (i < (b.Length & ~31))
                        {
                            d[j + 0] = Avx.Divide(aa, bb[j + 0]);
                            d[j + 1] = Avx.Divide(aa, bb[j + 1]);
                            d[j + 2] = Avx.Divide(aa, bb[j + 2]);
                            d[j + 3] = Avx.Divide(aa, bb[j + 3]);
                            i += 32;
                            j += 4;
                        }
                        while (i < (b.Length & ~7))
                        {
                            d[j] = Avx.Divide(aa, bb[j]);
                            i += 8;
                            j++;
                        }
                    }
                }
            }

            unchecked
            {
                while (i < (b.Length & ~3))
                {
                    output[i + 0] = a / b[i + 0];
                    output[i + 1] = a / b[i + 1];
                    output[i + 2] = a / b[i + 2];
                    output[i + 3] = a / b[i + 3];
                    i += 4;
                }
                while (i < b.Length)
                {
                    output[i] = a / b[i];
                    i++;
                }
            }

            return output;
        }

    }
}
