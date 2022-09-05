using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NSS
{
    public static class Span2DExtensions
    {
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span2D<T> AsSpan2D<T>(this Array array) where T : struct
        {
            Span<T> span = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(array)), array.Length);

            return new Span2D<T>(
                span,
                array.GetLength(0),
                array.GetLength(1)
            );
        }
    }
}
