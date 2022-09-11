using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Diagnostics.Contracts;

namespace NSS
{
    static public partial class Intrinsics
    {
        /// <summary>
        /// => make sure data is aligned 
        /// output = a * b + c
        /// </summary>
        public static Span<float> Add(Span<float> a, Span<float> b, Span<float> output)
        {
            int i = 0;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> bb = MemoryMarshal.Cast<float, Vector256<float>>(b);
                    Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(output);
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~31))
                        {
                            d[j + 0] = Avx.Add(aa[j + 0], bb[j + 0]);
                            d[j + 1] = Avx.Add(aa[j + 1], bb[j + 1]);
                            d[j + 2] = Avx.Add(aa[j + 2], bb[j + 2]);
                            d[j + 3] = Avx.Add(aa[j + 3], bb[j + 3]);
                            i += 32;
                            j += 4;
                        }
                        while (i < (a.Length & ~7))
                        {
                            d[j] = Avx.Add(aa[j], bb[j]);
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
                    output[i + 0] = a[i + 0] + b[i + 0];
                    output[i + 1] = a[i + 1] + b[i + 1];
                    output[i + 2] = a[i + 2] + b[i + 2];
                    output[i + 3] = a[i + 3] + b[i + 3];
                    i += 4;
                }
                while (i < a.Length)
                {
                    output[i] = a[i] + b[i];
                    i++;
                }
            }

            return output;
        }

        public static Span<float> Substract(Span<float> a, Span<float> b, Span<float> output)
        {
            int i = 0;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> bb = MemoryMarshal.Cast<float, Vector256<float>>(b);
                    Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(output);
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~31))
                        {
                            d[j + 0] = Avx.Subtract(aa[j + 0], bb[j + 0]);
                            d[j + 1] = Avx.Subtract(aa[j + 1], bb[j + 1]);
                            d[j + 2] = Avx.Subtract(aa[j + 2], bb[j + 2]);
                            d[j + 3] = Avx.Subtract(aa[j + 3], bb[j + 3]);
                            i += 32;
                            j += 4;
                        }
                        while (i < (a.Length & ~7))
                        {
                            d[j] = Avx.Subtract(aa[j], bb[j]);
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
                    output[i + 0] = a[i + 0] - b[i + 0];
                    output[i + 1] = a[i + 1] - b[i + 1];
                    output[i + 2] = a[i + 2] - b[i + 2];
                    output[i + 3] = a[i + 3] - b[i + 3];
                    i += 4;
                }
                while (i < a.Length)
                {
                    output[i] = a[i] - b[i];
                    i++;
                }
            }

            return output;
        }

        /// <summary>
        /// => make sure data is aligned 
        /// output = a * b + c
        /// </summary>
        public static Span<float> AddScalar(Span<float> a, float b, Span<float> output)
        {
            int i = 0;

            if (Avx.IsSupported)
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
                            d[j + 0] = Avx.Add(aa[j + 0], bb);
                            d[j + 1] = Avx.Add(aa[j + 1], bb);
                            d[j + 2] = Avx.Add(aa[j + 2], bb);
                            d[j + 3] = Avx.Add(aa[j + 3], bb);
                            i += 32;
                            j += 4;
                        }
                        while (i < (a.Length & ~7))
                        {
                            d[j] = Avx.Add(aa[j], bb);
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
                    output[i + 0] = a[i + 0] + b;
                    output[i + 1] = a[i + 1] + b;
                    output[i + 2] = a[i + 2] + b;
                    output[i + 3] = a[i + 3] + b;
                    i += 4;
                }
                while (i < a.Length)
                {
                    output[i] = a[i] + b;
                    i++;
                }
            }

            return output;
        }


        public static Span<float> SubstractScalar(Span<float> a, float b, Span<float> output)
        {
            int i = 0;

            if (Avx.IsSupported)
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
                            d[j + 0] = Avx.Subtract(aa[j + 0], bb);
                            d[j + 1] = Avx.Subtract(aa[j + 1], bb);
                            d[j + 2] = Avx.Subtract(aa[j + 2], bb);
                            d[j + 3] = Avx.Subtract(aa[j + 3], bb);
                            i += 32;
                            j += 4;
                        }
                        while (i < (a.Length & ~7))
                        {
                            d[j] = Avx.Subtract(aa[j], bb);
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
                    output[i + 0] = a[i + 0] - b;
                    output[i + 1] = a[i + 1] - b;
                    output[i + 2] = a[i + 2] - b;
                    output[i + 3] = a[i + 3] - b;
                    i += 4;
                }
                while (i < a.Length)
                {
                    output[i] = a[i] - b;
                    i++;
                }
            }

            return output;
        }
        public static Span<float> SubstractScalar(float a, Span<float> b, Span<float> output)
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
                            d[j + 0] = Avx.Subtract(aa, bb[j + 0]);
                            d[j + 1] = Avx.Subtract(aa, bb[j + 1]);
                            d[j + 2] = Avx.Subtract(aa, bb[j + 2]);
                            d[j + 3] = Avx.Subtract(aa, bb[j + 3]);
                            i += 32;
                            j += 4;
                        }
                        while (i < (b.Length & ~7))
                        {
                            d[j] = Avx.Subtract(aa, bb[j]);
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
                    output[i + 0] = a - b[i + 0];
                    output[i + 1] = a - b[i + 1];
                    output[i + 2] = a - b[i + 2];
                    output[i + 3] = a - b[i + 3];
                    i += 4;
                }
                while (i < b.Length)
                {
                    output[i] = a - b[i];
                    i++;
                }
            }

            return output;
        }


    }
}
