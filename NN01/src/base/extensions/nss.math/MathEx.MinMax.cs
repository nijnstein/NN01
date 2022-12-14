using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
	static public partial class MathEx
    {
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int Max(int a, int b, int c)
		{
			if (a > b)
			{
				if (c > a)
					return c;
				else
					return a;
			}
			else
			{
				if (c > b)
					return c;
				else
					return b;
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Max(double a, double b, double c)
		{
			if (a > b)
			{
				if (c > a)
					return c;
				else
					return a;
			}
			else
			{
				if (c > b)
					return c;
				else
					return b;
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static float Max(float a, float b, float c)
		{
			if (a > b)
			{
				if (c > a)
					return c;
				else
					return a;
			}
			else
			{
				if (c > b)
					return c;
				else
					return b;
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static float Min(int a, int b, int c)
		{
			if (a < b)
			{
				if (c < a)
					return c;
				else
					return a;
			}
			else
			{
				if (c < b)
					return c;
				else
					return b;
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Min(double a, double b, double c)
		{
			if (a < b)
			{
				if (c < a)
					return c;
				else
					return a;
			}
			else
			{
				if (c < b)
					return c;
				else
					return b;
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static float Min(float a, float b, float c)
		{
			if (a < b)
			{
				if (c < a)
					return c;
				else
					return a;
			}
			else
			{
				if (c < b)
					return c;
				else
					return b;
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int Min(int a, int b, int c, int d)
		{
			if (a < b)
			{
				if (c < a)
				{
					if (d < c)
						return d;
					else
						return c;
				}
				else
				{
					if (d < a)
						return d;
					else
						return a;
				}
			}
			else
			{
				if (c < b)
				{
					if (d < c)
						return d;
					else
						return c;
				}
				else
				{
					if (d < b)
						return d;
					else
						return b;
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static uint Min(uint a, uint b, uint c, uint d)
		{
			if (a < b)
			{
				if (c < a)
				{
					if (d < c)
						return d;
					else
						return c;
				}
				else
				{
					if (d < a)
						return d;
					else
						return a;
				}
			}
			else
			{
				if (c < b)
				{
					if (d < c)
						return d;
					else
						return c;
				}
				else
				{
					if (d < b)
						return d;
					else
						return b;
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Min(double a, double b, double c, double d)
		{
			if (a < b)
			{
				if (c < a)
				{
					if (d < c)
						return d;
					else
						return c;
				}
				else
				{
					if (d < a)
						return d;
					else
						return a;
				}
			}
			else
			{
				if (c < b)
				{
					if (d < c)
						return d;
					else
						return c;
				}
				else
				{
					if (d < b)
						return d;
					else
						return b;
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static float Min(float a, float b, float c, float d)
		{
			if (a < b)
			{
				if (c < a)
				{
					if (d < c)
						return d;
					else
						return c;
				}
				else
				{
					if (d < a)
						return d;
					else
						return a;
				}
			}
			else
			{
				if (c < b)
				{
					if (d < c)
						return d;
					else
						return c;
				}
				else
				{
					if (d < b)
						return d;
					else
						return b;
				}
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int Max(int a, int b, int c, int d)
		{
			if (a > b)
			{
				if (c > a)
				{
					if (d > c)
						return d;
					else
						return c;
				}
				else
				{
					if (d > a)
						return d;
					else
						return a;
				}
			}
			else
			{
				if (c > b)
				{
					if (d > c)
						return d;
					else
						return c;
				}
				else
				{
					if (d > b)
						return d;
					else
						return b;
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static uint Max(uint a, uint b, uint c, uint d)
		{
			if (a > b)
			{
				if (c > a)
				{
					if (d > c)
						return d;
					else
						return c;
				}
				else
				{
					if (d > a)
						return d;
					else
						return a;
				}
			}
			else
			{
				if (c > b)
				{
					if (d > c)
						return d;
					else
						return c;
				}
				else
				{
					if (d > b)
						return d;
					else
						return b;
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Max(double a, double b, double c, double d)
		{
			if (a > b)
			{
				if (c > a)
				{
					if (d > c)
						return d;
					else
						return c;
				}
				else
				{
					if (d > a)
						return d;
					else
						return a;
				}
			}
			else
			{
				if (c > b)
				{
					if (d > c)
						return d;
					else
						return c;
				}
				else
				{
					if (d > b)
						return d;
					else
						return b;
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static float Max(float a, float b, float c, float d)
		{
			if (a > b)
			{
				if (c > a)
				{
					if (d > c)
						return d;
					else
						return c;
				}
				else
				{
					if (d > a)
						return d;
					else
						return a;
				}
			}
			else
			{
				if (c > b)
				{
					if (d > c)
						return d;
					else
						return c;
				}
				else
				{
					if (d > b)
						return d;
					else
						return b;
				}
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void MinMax(double[] da, out double min, out double max)
		{
			min = double.MaxValue;
			max = double.MinValue;

			if (da == null || da.Length == 0) return;

			unchecked
			{
				for (int i = 0; i < da.Length; i++)
				{
					double d = da[i];
					if (d < min) min = d;
					if (d > max) max = d;
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void MinMax(Span<float> a, out float min, out float max)
		{
			min = float.MaxValue;
			max = float.MinValue;

			if (a == null || a.Length == 0) return;

			unchecked
			{
				for (int i = 0; i < a.Length; i++)
				{
					float f = a[i];
					if (f < min) min = f;
					if (f > max) max = f;
				}
			}
		}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMax(this Span<float> f)
        {
            float max = float.MinValue;
            int j = -1;
            unchecked
            {
                for (int i = 0; i < f.Length; i++)
                {
                    if (f[i] > max)
                    {
                        max = f[i];
                        j = i;
                    }
                }
            }
            return j;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMax(this float[] f)
        {
            return ArgMax(f.AsSpan());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMin(this Span<float> f)
        {
            float min = float.MaxValue;
            int j = -1;
            unchecked
            {
                for (int i = 0; i < f.Length; i++)
                {
                    if (f[i] < min)
                    {
                        min = f[i];
                        j = i;
                    }
                }
            }
            return j;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMin(this float[] f)
        {
            return ArgMin(f.AsSpan());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMax(this ReadOnlySpan<float> f)
        {
            float max = float.MinValue;
            int j = -1;
            unchecked
            {
                for (int i = 0; i < f.Length; i++)
                {
                    if (f[i] > max)
                    {
                        max = f[i];
                        j = i;
                    }
                }
            }
            return j;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ArgMin(this ReadOnlySpan<float> f)
        {
            float min = float.MaxValue;
            int j = -1;
            unchecked
            {
                for (int i = 0; i < f.Length; i++)
                {
                    if (f[i] < min)
                    {
                        min = f[i];
                        j = i;
                    }
                }
            }
            return j;
        }
    }
}
