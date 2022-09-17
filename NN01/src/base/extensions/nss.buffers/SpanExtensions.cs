using ILGPU.IR;
using System.Diagnostics;
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

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span2D<T> AsSpan2D<T>(this Array array, int rows, int columns) where T : struct
        {
            Debug.Assert(array.Length == rows * columns, "input array size mismatches with given row and columncount");

            Span<T> span = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(array)), array.Length);

            return new Span2D<T>(
                span,
                rows,
                columns
            );
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span3D<T> AsSpan3D<T>(this Array array) where T : struct
        {
            Span<T> span = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(array)), array.Length);

            return new Span3D<T>(
                span,
                array.GetLength(0),
                array.GetLength(1),
                array.GetLength(2)
            );
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Zero(this Span<float> a)
        {
            unchecked
            {
                int i = 0;
                while (i < (a.Length & ~7))
                {
                    a[i + 0] = 0f;
                    a[i + 1] = 0f;
                    a[i + 2] = 0f;
                    a[i + 3] = 0f;

                    a[i + 4] = 0f;
                    a[i + 5] = 0f;
                    a[i + 6] = 0f;
                    a[i + 7] = 0f;

                    i += 8;
                }
                while (i < a.Length)
                {
                    a[i++] = 0f;
                }
            }
            return a;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<int> Zero(this Span<int> a)
        {
            unchecked
            {
                int i = 0;
                while (i < (a.Length & ~7))
                {
                    a[i + 0] = 0;
                    a[i + 1] = 0;
                    a[i + 2] = 0;
                    a[i + 3] = 0;

                    a[i + 4] = 0;
                    a[i + 5] = 0;
                    a[i + 6] = 0;
                    a[i + 7] = 0;

                    i += 8;
                }
                while (i < a.Length)
                {
                    a[i++] = 0;
                }
            }
            return a;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Ones(this Span<float> a)
        {
            unchecked
            {
                int i = 0;
                while (i < (a.Length & ~7))
                {
                    a[i + 0] = 1f;
                    a[i + 1] = 1f;
                    a[i + 2] = 1f;
                    a[i + 3] = 1f;
                    a[i + 4] = 1f;
                    a[i + 5] = 1f;
                    a[i + 6] = 1f;
                    a[i + 7] = 1f;
                    i += 8;
                }
                while (i < a.Length)
                {
                    a[i++] = 1f;
                }
            }
            return a;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Fill(this Span<float> a, float f)
        {
            unchecked
            {
                int i = 0;
                while (i < (a.Length & ~7))
                {
                    a[i + 0] = f;
                    a[i + 1] = f;
                    a[i + 2] = f;
                    a[i + 3] = f;
                    a[i + 4] = f;
                    a[i + 5] = f;
                    a[i + 6] = f;
                    a[i + 7] = f;
                    i += 8;
                }
                while (i < a.Length)
                {
                    a[i++] = f;
                }
            }
            return a;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<int> Fill(this Span<int> a, int f)
        {
            unchecked
            {
                int i = 0;
                while (i < (a.Length & ~7))
                {
                    a[i + 0] = f;
                    a[i + 1] = f;
                    a[i + 2] = f;
                    a[i + 3] = f;
                    a[i + 4] = f;
                    a[i + 5] = f;
                    a[i + 6] = f;
                    a[i + 7] = f;
                    i += 8;
                }
                while (i < a.Length)
                {
                    a[i++] = f;
                }
            }
            return a;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<uint> Fill(this Span<uint> a, uint f)
        {
            unchecked
            {
                int i = 0;
                while (i < (a.Length & ~7))
                {
                    a[i + 0] = f;
                    a[i + 1] = f;
                    a[i + 2] = f;
                    a[i + 3] = f;
                    a[i + 4] = f;
                    a[i + 5] = f;
                    a[i + 6] = f;
                    a[i + 7] = f;
                    i += 8;
                }
                while (i < a.Length)
                {
                    a[i++] = f;
                }
            }
            return a;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsBitSet(this Span<int> a, int bit_index)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);
            byte mask = (byte)(1 << (byte)index);
            return (a[index] & mask) == mask;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EnableBit(this Span<int> a, int bit_index)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);
            int mask = 1 << index;
            a[index] = a[index] | mask;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void DisableBit(this Span<int> a, int bit_index)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);
            int mask = 1 << index;
            a[index] = a[index] & ~mask;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetBit(this Span<int> a, int bit_index, bool value)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);
            int mask = 1 << index;
            a[index] = value ? a[index] | mask : a[index] & ~mask;
        }

   
    }
}
