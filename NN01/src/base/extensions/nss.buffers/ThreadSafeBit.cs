using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static class ThreadSafeBit
    {
        /// <summary>
        /// check if the nth bit is set
        /// </summary>
        /// <param name="temp">some memory</param>
        /// <param name="index">zero based index in bits</param>
        /// <returns></returns>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsSet(byte data, byte index)
        {
            byte mask = (byte)(1 << (byte)index);
            return (data & mask) == mask;
        }

        /// <summary>
        /// check if a bit is set
        /// </summary>
        /// <param name="data"></param>
        /// <param name="index">zero based bit index</param>
        /// <returns></returns>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsSet(uint data, byte index)
        {
            uint mask = (uint)(1 << index);
            return (data & mask) == mask;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsSet(ulong data, byte index)
        {
            ulong mask = ((ulong)1 << index);
            return (data & mask) == mask;
        }

        /// <summary>
        /// threadsafe set-bit 
        /// </summary>
        /// <param name="bit"></param>
        /// <param name="bitmap"></param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static public void EnableBitSafe(int bit_index, Span<int> bitmap, int attempts = 1024)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);

            int i = 0;

            do
            {
                int current = bitmap[index];

                // check if set 
                if ((current & bit) == bit) return;

                int next = current | bit;

                int value = Interlocked.CompareExchange(ref bitmap[index], next, current);
                if (value == current)
                {
                    // value set, it didnt change meanwhile 
                    return;
                }

                // value was changed while trying to set it, restart procedure 
            }
            while (i++ < attempts);
        }

        /// <summary>
        /// threadsafe set-bit 
        /// </summary>
        /// <param name="bit"></param>
        /// <param name="bitmap"></param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static public void SetBitSafe(int bit_index, bool bit_value, Span<int> bitmap, int attempts = 1024)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);

            int i = 0;

            do
            {
                int current = bitmap[index];

                // check if set 
                if ((current & bit) == bit) return;

                int next = bit_value ? current | bit : current & ~bit;

                int value = Interlocked.CompareExchange(ref bitmap[index], next, current);
                if (value == current)
                {
                    // value set, it didnt change meanwhile 
                    return;
                }

                // value was changed while trying to set it, restart procedure 
            }
            while (i++ < attempts);
        }
        
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static public void DisableBitSafe(int bit_index, Span<int> bitmap, int attempts = 1024)
        {
            int index = bit_index >> 5;
            int bit = 1 << (bit_index & 31);

            int i = 0;

            do
            {
                int current = bitmap[index];

                // check if set 
                if ((current & bit) == bit) return;

                int next = current & ~bit;

                int value = Interlocked.CompareExchange(ref bitmap[index], next, current);
                if (value == current)
                {
                    // value set, it didnt change meanwhile 
                    return;
                }

                // value was changed while trying to set it, restart procedure 
            }
            while (i++ < attempts);
        }


        /// <summary>
        /// set nth bit of memory to value
        /// </summary>
        /// <param name="temp">some memory</param>
        /// <param name="index">index in bits</param>
        /// <param name="value"></param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetBitUnsafe(ref byte data, byte index, bool value)
        {
            byte mask = (byte)(1 << (byte)index);
            data = value ? (byte)(data | mask) : (byte)(data & ~mask);
        }


        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetBitUnsafe(ref uint data, byte index, bool value)
        {
            uint mask = (uint)(1 << index);
            data = value ? (uint)(data | mask) : (uint)(data & ~mask);
        }
    }
}
