using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NSS
{
    public class BitBuffer2D : BitBuffer
    {
        protected int width;
        protected int height;
        public int BitWidth => width; 
        public int BitHeight => height; 

        public BitBuffer2D(int bitCountWidth, int bitCountHeight) : base(bitCountWidth * bitCountHeight)
        {
            width = bitCountWidth;
            height = bitCountHeight;
        }

        public BitBuffer2D(int bitCountWidth, int bitCountHeight, int[] buffer) : base(bitCountWidth * bitCountHeight, buffer)
        {
            width = bitCountWidth;
            height = bitCountHeight;
        }

        public bool this[int x, int y] => this[x + y * BitWidth]; 

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsSet(int x, int y) => buffer.IsBitSet(x + y * width);
   
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set(int x, int y, bool value) => buffer.SetBit(x + y * width, value);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Enable(int x, int y) => buffer.SetBit(x + y * width, true);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Disable(int x, int y) => buffer.SetBit(x + y * width, false);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetSafe(int x, int y, bool value, int attempts = 1024)
        {
            int bit = x + y * width;
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
        public void EnableSafe(int x, int y, int attempts = 1024) => ThreadSafeBit.EnableBitSafe(x + y * width, buffer, attempts);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void DisableSafe(int x, int y, int attempts = 1024) => ThreadSafeBit.DisableBitSafe(x + y * width, buffer);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AnySetInRow(int y)
        {
            // TODO    big performance optimizations posible here... 
            for (int x = 0; x < BitWidth; x++)
            {
                if (IsSet(x, y)) return true;
            }
            return false;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool NoneSetInRow(int y)
        {
            // TODO    big performance optimizations posible here... 
            for (int x = 0; x < BitWidth; x++)
            {
                if (IsSet(x, y)) return false;
            }
            return true;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AnySetInColumn(int x)
        {
            // TODO    big performance optimizations posible here... 
            for (int y = 0; y < BitHeight; y++)
            {
                if (IsSet(x, y)) return true;
            }
            return false;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool NoneSetInColumn(int x)
        {
            // TODO    big performance optimizations posible here... 
            for (int y = 0; y < BitHeight; y++)
            {
                if (IsSet(x, y)) return false;
            }
            return true;
        }

        /// <summary>
        /// slice buffer on row and transform into a one-hot encoded float vector
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<float> SliceRowToVector256(int i, int j)
        {
            return Vector256.Create(
               IsSet(i, j + 0) ? 1f : 0f,
               IsSet(i, j + 1) ? 1f : 0f,
               IsSet(i, j + 2) ? 1f : 0f,
               IsSet(i, j + 3) ? 1f : 0f,
               IsSet(i, j + 4) ? 1f : 0f,
               IsSet(i, j + 5) ? 1f : 0f,
               IsSet(i, j + 6) ? 1f : 0f,
               IsSet(i, j + 7) ? 1f : 0f
            );
        }

        /// <summary>
        /// slice buffer on column and transform into a one-hot encoded float vector
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<float> SliceColumnToVector256(int i, int j)
        {
            return Vector256.Create(
               IsSet(i + 0, j) ? 1f : 0f,
               IsSet(i + 1, j) ? 1f : 0f,
               IsSet(i + 2, j) ? 1f : 0f,
               IsSet(i + 3, j) ? 1f : 0f,
               IsSet(i + 4, j) ? 1f : 0f,
               IsSet(i + 5, j) ? 1f : 0f,
               IsSet(i + 6, j) ? 1f : 0f,
               IsSet(i + 7, j) ? 1f : 0f
            );
        }
    }
}
