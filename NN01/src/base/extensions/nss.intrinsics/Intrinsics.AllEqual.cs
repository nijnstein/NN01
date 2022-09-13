using ILGPU.IR.Values;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
	public static partial class Intrinsics
	{
		public static bool AllEqual(Span<float> input)
		{
			Debug.Assert(input != null && input.Length > 0);

			int i = 0;
			bool equal = true; 

			if (Avx2.IsSupported)
			{
				Span<Vector256<float>> p256 = MemoryMarshal.Cast<float, Vector256<float>>(input);

				// fill v256 with all the first value 
				Vector256<float> pop_0th = Vector256.Create(input[0]);

				// compare 64 floats as 1111 or 0000 to 256 bits of 1							 
				while ((i < (input.Length & ~63)) & equal)
				{
					// shuffle 1111 into each nibble for each match
					Vector256<int> a = Vector256.Create(
					    Avx2.MoveMask(Avx2.CompareEqual(p256[0], pop_0th)),
						Avx2.MoveMask(Avx2.CompareEqual(p256[1], pop_0th)),
						Avx2.MoveMask(Avx2.CompareEqual(p256[2], pop_0th)),
						Avx2.MoveMask(Avx2.CompareEqual(p256[3], pop_0th)),
						Avx2.MoveMask(Avx2.CompareEqual(p256[4], pop_0th)),
						Avx2.MoveMask(Avx2.CompareEqual(p256[5], pop_0th)),
						Avx2.MoveMask(Avx2.CompareEqual(p256[6], pop_0th)),
						Avx2.MoveMask(Avx2.CompareEqual(p256[7], pop_0th)));

					// equal if 256i == all ones 
					equal = ((uint)Avx2.MoveMask(Avx2.CompareEqual(a, v256i_one).AsByte()) == 0xFFFFFFFF);

					i += 64;
				}
			}
			while ((i < input.Length) & equal)
			{
				equal = input[0] == input[i];
			}
			return equal;
		}

		public static bool AllEqual(Span<int> input)
		{
			Debug.Assert(input != null && input.Length > 0);

			int i = 0;
			bool equal = true;

			if (Avx2.IsSupported)
			{
				Span<Vector256<int>> p256 = MemoryMarshal.Cast<int, Vector256<int>>(input);

				// fill v256 with all the first value 
				Vector256<int> pop_0th = Vector256.Create(input[0]);

				// compare 64 floats as 1111 or 0000 to 256 bits of 1							 
				while ((i < (input.Length & ~63)) & equal)
				{
					// shuffle 1111 into each nibble for each match
					Vector256<int> a = Vector256.Create(
						Avx2.MoveMask(Avx2.CompareEqual(p256[0], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[1], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[2], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[3], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[4], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[5], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[6], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[7], pop_0th).AsByte()));

					// equal if 256i == all ones 
					equal = ((uint)Avx2.MoveMask(Avx2.CompareEqual(a, v256i_one).AsByte()) == 0xFFFFFFFF);
					i += 64;
				}
			}
			while ((i < input.Length) & equal)
			{
				equal = input[0] == input[i];
			}
			return equal;
		}

		public static bool AllEqual(Span<uint> input)
		{
			Debug.Assert(input != null && input.Length > 0);

			int i = 0;
			bool equal = true;

			if (Avx2.IsSupported)
			{
				Span<Vector256<uint>> p256 = MemoryMarshal.Cast<uint, Vector256<uint>>(input);

				// fill v256 with all the first value 
				Vector256<uint> pop_0th = Vector256.Create(input[0]);

				// compare 64 floats as 1111 or 0000 to 256 bits of 1							 
				while ((i < (input.Length & ~63)) & equal)
				{
					// shuffle 1111 into each nibble for each match
					Vector256<int> a = Vector256.Create(
						Avx2.MoveMask(Avx2.CompareEqual(p256[0], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[1], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[2], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[3], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[4], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[5], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[6], pop_0th).AsByte()),
						Avx2.MoveMask(Avx2.CompareEqual(p256[7], pop_0th).AsByte()));

					// equal if 256i == all ones 
					equal = ((uint)Avx2.MoveMask(Avx2.CompareEqual(a, v256i_one).AsByte()) == 0xFFFFFFFF);
					i += 64;
				}
			}
			while ((i < input.Length) & equal)
			{
				equal = input[0] == input[i];
			}
			return equal;
		}


	}
}
