using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NSS
{
    public static class SpanExtensions
    {
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span<T> AsSpan<T>(this Array array)
        {
            return MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(array)), array.Length);
        }            
    }
}
