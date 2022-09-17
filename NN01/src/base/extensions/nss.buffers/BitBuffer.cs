using ILGPU.Algorithms;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace NSS
{
    public class BitBuffer : IDisposable
    {
        public readonly int BitCount;

        protected bool bufferIsLeased;
        protected int[] buffer;

        public BitBuffer(int bitCount)
        {
            Debug.Assert(bitCount > 0);
            BitCount = bitCount;
            bufferIsLeased = true;
            buffer = ArrayPool<int>.Shared.Rent((BitCount & ~31) + 1);
        }
        public BitBuffer(int bitCount, int[] bufferBase)
        {
            Debug.Assert(bitCount > 0);
            Debug.Assert(bufferBase != null);
            Debug.Assert(bufferBase.Length >= (BitCount & ~31) + 1);

            BitCount = bitCount;
            bufferIsLeased = false;
            buffer = ArrayPool<int>.Shared.Rent((BitCount & ~31) + 1);
        }
        public bool this[int bitIndex]
        {
            get
            {
                return IsSet(bitIndex);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ZeroAll() => DisableAll();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void DisableAll()
        {
            buffer.Zero(); 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void EnableAll() => buffer.Fill(-1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsSet(int bit)
        {
            return buffer.IsBitSet(bit);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set(int bit, bool value)
        {
            buffer.SetBit(bit, value); 
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Enable(int bit)
        {
            buffer.SetBit(bit, true);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Disable(int bit)
        {
            buffer.SetBit(bit, false);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetSafe(int bit, bool value, int attempts = 1024)
        {
            if (value)
            {
                ThreadSafeBit.EnableBitSafe(bit, buffer, attempts);
            }
            else
            {
                ThreadSafeBit.DisableBitSafe(bit, buffer, attempts);
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void EnableSafe(int bit, int attempts = 1024)
        {
            ThreadSafeBit.EnableBitSafe(bit, buffer, attempts);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void DisableSafe(int bit, int attempts = 1024)
        {
            ThreadSafeBit.DisableBitSafe(bit, buffer);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AnySet()
        {
            unchecked
            {
                for(int i = 0; i < buffer.Length; i++)
                {
                    if (buffer[i] != 0) return true; 
                }
                return false;
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool NoneSet() => !AnySet();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long CountSetLong()
        {
            long sum = 0;
            unchecked
            {
                for (int i = 0; i < buffer.Length; i++)
                {
                    sum += MathEx.CountBits((uint)buffer[i]);
                }
            }
            return sum;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long CountSet()
        {
            int sum = 0;
            unchecked
            {
                for (int i = 0; i < buffer.Length; i++)
                {
                    sum += MathEx.CountBits((uint)buffer[i]);
                }
            }
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long CountUnSet() => BitCount - CountSet();

        public void Dispose()
        {
            ArrayPool<int>.Shared.Return(buffer); 
        }
    }
}
