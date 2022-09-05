using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static partial class MathEx
    {
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static int HighestBitSet(uint n)
		{
			return 63 - System.Numerics.BitOperations.LeadingZeroCount(n);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static int LowestBitSet(uint n)
		{
			return System.Numerics.BitOperations.TrailingZeroCount(n);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static int HighestBitSet(ulong n)
		{
			return 63 - System.Numerics.BitOperations.LeadingZeroCount(n);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static int LowestBitSet(ulong n)
		{
			return System.Numerics.BitOperations.TrailingZeroCount(n);
		}

		static public byte[] BitReverseTable =
  {
  0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0,
  0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
  0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
  0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
  0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4,
  0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
  0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
  0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
  0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
  0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
  0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea,
  0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
  0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6,
  0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
  0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
  0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
  0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1,
  0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
  0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9,
  0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
  0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
  0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
  0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed,
  0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
  0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3,
  0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
  0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
  0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
  0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7,
  0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
  0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
  0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff
  };

		static public byte Reverse(byte b)
		{
			return BitReverseTable[b];
		}

		static public uint Reverse(uint n)
		{
			uint r = 0;
			byte t = (byte)(n >> 24);
			// could optimize by removing the shifts on the bitreversetable by using 4 tables
			// with the values pre-shifted, that would require 4x256x4 (32bits) bytes which ain't 
			// good for the cache.. though modern processors have a lot.. need to test TODO
			r |= BitReverseTable[t];
			t = (byte)(n >> 16 & 0x000000FF);
			r |= (uint)(BitReverseTable[t] << 8);
			t = (byte)(n >> 8 & 0x000000FF);
			r |= (uint)(BitReverseTable[t] << 16);
			t = (byte)(n & 0x000000FF);
			r |= (uint)(BitReverseTable[t] << 24);
			return r;
		}


		/// <summary>
		/// ABCDEFGH = GHEFCDAB
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static public uint ReverseByteOrder(uint u)
		{
			// ABCDEFGH = GHEFCDAB
			return ((u & 0x000000FF) << 24) +
				   ((u & 0x0000FF00) << 8) +
				   ((u & 0x00FF0000) >> 8) +
				   ((u & 0xFF000000) >> 24);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int CountLeadingZeros(this ulong input)
		{
#if NETCOREAPP3_1_OR_GREATER
            return System.Numerics.BitOperations.LeadingZeroCount(input);
#else
			const int bits = 64;
			if (input == 0L) return bits; // Not needed. Use only if 0 is very common.

			int n = 1;
			if ((input >> (bits - 32)) == 0) { n += 32; input <<= 32; }
			if ((input >> (bits - 16)) == 0) { n += 16; input <<= 16; }
			if ((input >> (bits - 8)) == 0) { n += 8; input <<= 8; }
			if ((input >> (bits - 4)) == 0) { n += 4; input <<= 4; }
			if ((input >> (bits - 2)) == 0) { n += 2; input <<= 2; }

			return n - (int)(input >> (bits - 1));
#endif
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int CountBits(uint i)
		{
#if NETCOREAPP3_1_OR_GREATER
			return System.Numerics.BitOperations.PopCount(i);
#else
			i = i - ((i >> 1) & 0x55555555);
			i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
			return (int)(((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
		}


		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int CountBits(ulong input)
		{
#if NETCOREAPP3_1_OR_GREATER
			return System.Numerics.BitOperations.PopCount(input);
#else
			return CountBits((uint)(input & 0x00000000FFFFFFFF)) + CountBits((uint)((input >> 32) & 0x00000000FFFFFFFF)); 
#endif
		}


		/// <summary>
		/// return if input is a power of 2 and returns the exponent in
		/// output. If input is not a power of 2 the nearest exponent is returned; 
		/// </summary>
		/// <param name="input">a number</param>
		/// <param name="output">the output power</param>
		/// <returns></returns>
		public static bool PowerOf2(uint input, out uint output)
		{
			if (IsPowerOf2(input))
			{
				output = (uint)HighestBitSet(input) - 1;
				return true;
			}
			else
			{
				output = (uint)HighestBitSet(NearestPowerOf2(input)) - 1;
				return false;
			}
		}

		/// <summary>
		/// return if input is a power of 2
		/// </summary>
		/// <param name="input">a number</param>
		/// <returns>true if input is a power of 2</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsPowerOf2(uint input)
		{
#if NETCOREAPP3_1_OR_GREATER
			return System.Numerics.BitOperations.IsPow2(input);
#else
			return (input & (input - 1)) == 0;
#endif
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsPowerOf2(int input)
		{
#if NETCOREAPP3_1_OR_GREATER
			return System.Numerics.BitOperations.IsPow2(input);
#else
			return (input & (input - 1)) == 0;
#endif
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsPowerOf2(ulong input)
		{
#if NETCOREAPP3_1_OR_GREATER
			return System.Numerics.BitOperations.IsPow2(input);
#else
			return (input & (input - 1)) == 0;
#endif
		}

		/// <summary>
		/// get the nearest power of two for the input number
		/// </summary>
		/// <param name="input">a number</param>
		/// <returns>nearest power of 2 for the input</returns>
		public static uint NearestPowerOf2(uint input)
		{
			uint u = input > 0 ? input - 1 : 0;
			u = u | (u >> 1);
			u = u | (u >> 2);
			u = u | (u >> 4);
			u = u | (u >> 8);
			u = u | (u >> 16);
			u = u + 1;
			return u;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static public int RoundUpToPowerOf2(this int n, int roundTo)
		{
			return (n + roundTo - 1) & -roundTo;
		}
		
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static public byte RoundUpToPowerOf2(this byte n, byte roundTo)
		{
			return (byte)((byte)(n + roundTo - 1) & (byte)-roundTo);
		}



	}
}
