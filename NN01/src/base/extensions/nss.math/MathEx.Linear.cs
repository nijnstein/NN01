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
        public static float Lerp(this float a, float b, float by)
        {
            return a + (b - a) * by;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Lerp(this double a, double b, double by)
        {
            return a + (b - a) * by;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CubicInterpolation(float v0, float v1, float v2, float v3, float t)
        {
            //var v01 = Lerp( v0, v1, t );
            //var v12 = Lerp( v1, v2, t );
            //var v23 = Lerp( v2, v3, t );
            //var v012 = Lerp( v01, v12, t );
            //var v123 = Lerp( v12, v23, t );
            //return Lerp( v012, v123, t );
            var p = (v3 - v2) - (v0 - v1);
            var q = (v0 - v1) - p;
            var r = v2 - v0;
            var s = v1;
            return (p * t * 3) + (q * t * 2) + (r * t) + s;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float QuadraticInterpolation(float v0, float v1, float v2, float t)
        {
            var v01 = Lerp(v0, v1, t);
            var v12 = Lerp(v1, v2, t);
            return Lerp(v01, v12, t);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CosInterpolation(float t)
        {
            t = -MathF.Cos(t * MathF.PI); // [-1, 1]
            return (t + 1) / 2; // [0, 1]
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float PerlinSmoothStep(float t)
        {
            // Ken Perlin's version
            return t * t * t * ((t * ((6 * t) - 15)) + 10);
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothStep(float t)
        {
            return t * t * (3 - (2 * t));
        }


        /// <summary>
        /// Calculates the approximate line y(x) = a + b(x) through the collection of points
        /// </summary>
        /// <param name="nx">x coords</param>
        /// <param name="ny">y coords</param>
        /// <param name="intercept">a</param>
        /// <param name="slope">b</param>
        /// <returns>the angle in degrees = (180 * Math.Atan(slope)) / Math.PI)</returns>
        public static double LinearRegression(int[] nx, int[] ny, out double intercept, out double slope)
        {
            slope = 0.0;
            intercept = 0.0;

            int n = nx.Count();

            int[] nxy = new int[n];
            for (int i = 0; i < n; i++) nxy[i] = nx[i] * ny[i];

            int[] nx2 = new int[n];
            for (int i = 0; i < n; i++) nx2[i] = nx[i] * nx[i];

            int sum_x = 0;
            int sum_y = 0;
            int sum_xy = 0;
            int sum_x2 = 0;

            for (int i = 0; i < n; i++)
            {
                sum_x += nx[i];
                sum_y += ny[i];
                sum_xy += nxy[i];
                sum_x2 += nx2[i];
            }

            if (sum_xy == 0) return 0.0;

            slope = ((double)n * sum_xy - sum_x * sum_y) / ((double)n * sum_x2 - sum_x * sum_x);
            intercept = ((double)sum_y - (slope * sum_x)) / (double)n;

            double angle = (180 * Math.Atan(slope)) / Math.PI;
            return angle;
        }

        public static float LinearRegression(int[] nx, int[] ny, out float intercept, out float slope)
        {
            slope = 0f;
            intercept = 0f;

            int n = nx.Count();

            int[] nxy = new int[n];
            for (int i = 0; i < n; i++) nxy[i] = nx[i] * ny[i];

            int[] nx2 = new int[n];
            for (int i = 0; i < n; i++) nx2[i] = nx[i] * nx[i];

            int sum_x = 0;
            int sum_y = 0;
            int sum_xy = 0;
            int sum_x2 = 0;

            for (int i = 0; i < n; i++)
            {
                sum_x += nx[i];
                sum_y += ny[i];
                sum_xy += nxy[i];
                sum_x2 += nx2[i];
            }

            if (sum_xy == 0) return 0f;

            slope = ((float)n * sum_xy - sum_x * sum_y) / ((float)n * sum_x2 - sum_x * sum_x);
            intercept = ((float)sum_y - (slope * sum_x)) / (float)n;

            float angle = (180 * MathF.Atan(slope)) / MathF.PI;
            return angle;
        }


    }
}
