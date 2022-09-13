using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static class ArrayUtils
    {

        public static T[,] ConvertTo2D<T>(this T[][] a)
        {
            Debug.Assert(a != null && a.Length > 0 && a[0].Length > 0); 
            Debug.Assert(a.All(x => x.Length == a[0].Length)); // dont skip the first .. collection of 1 

            int rows = a.Length;
            int columns = a[0].Length;

            T[,] b = new T[rows, columns];

            unchecked
            {
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < columns; c++)
                        b[r, c] = a[r][c];
            }

            return b; 
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsBitSet(this int[] a, int bit_index)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);
            byte mask = (byte)(1 << (byte)index);
            return (a[index] & mask) == mask;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EnableBit(this int[] a, int bit_index)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);
            int mask = 1 << index;
            a[index] = a[index] | mask;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void DisableBit(this int[] a, int bit_index)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);
            int mask = 1 << index;
            a[index] = a[index] & ~mask;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetBit(this int[] a, int bit_index, bool value)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);
            int mask = 1 << index;
            a[index] = value ? a[index] | mask : a[index] & ~mask;
        }

    }
}
