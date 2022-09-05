using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Diagnostics.Contracts;
using System.Diagnostics;

namespace NSS
{
    static public partial class Intrinsics
    {

        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static float AverageSquaredDifferences(Span<float> a, Span<float> b)
        {
            Debug.Assert(a != null);
            Debug.Assert(b != null);
            Debug.Assert(a.Length > 0);
            return Intrinsics.SumSquaredDifferences(a, b) / a.Length;
        }

        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static float Average(Span<float> a)
        {
            Debug.Assert(a != null);
            Debug.Assert(a.Length > 0); 
            return Intrinsics.Sum(a) / a.Length;
        }

    }
}
