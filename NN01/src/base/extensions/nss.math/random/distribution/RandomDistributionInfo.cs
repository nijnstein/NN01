using ILGPU.Algorithms.Random;
using ILGPU.IR.Analyses;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using static ILGPU.IR.Analyses.Uniforms;

namespace NSS
{


    /// <summary>
    /// information about the distribution to generate with GPURandomProvider
    /// </summary>
    public struct RandomDistributionInfo
    {
        public RandomDistributionType DistributionType;
        public RandomProviderBackend ProviderBackend;

        public float P1;
        public float P2;
        public float P3;

        public int RandomInputCount;

        public Func<SimpleRNG, float, float, float, float> Kernel;

        public float Mean
        {
            get
            {
                Debug.Assert(DistributionType != RandomDistributionType.Uniform);
                return P1;
            }
        }
        public float SD
        {
            get
            {
                Debug.Assert(DistributionType != RandomDistributionType.Uniform);
                return P2;
            }
        }
        public float Fan
        {
            get
            {
                Debug.Assert(DistributionType == RandomDistributionType.HeNormal);
                return P3;
            }
        }
        public float Low
        {
            get
            {
                Debug.Assert(DistributionType == RandomDistributionType.Uniform);
                return P1;
            }
        }
        public float High
        {
            get
            {
                Debug.Assert(DistributionType == RandomDistributionType.Uniform);
                return P2;
            }
        }

        public static RandomDistributionInfo Default => Uniform(0f, 1f);
        public static RandomProviderBackend DefaultProviderBackend => RandomProviderBackend.XorShift32;

        /// <summary>
        /// uniform distribution at mean p1 with sd p2 
        /// </summary>
        /// <param name="r">random [01)</param>
        /// <param name="p1">mean</param>
        /// <param name="p2">sd</param>
        /// <param name="p3">not used</param>

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float UniformKernel(SimpleRNG rng, float low, float high, float p3)
            => (high - low) * rng.GetUniformSingle() + low;
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LogNormalKernel(SimpleRNG rng, float p1, float p2, float p3)
            => rng.GetLogNormalSingle(p1, p2);  // MathEx.Gaussian(Random.Shared, p1, p2);
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalKernel(SimpleRNG rng, float p1, float p2, float p3)
            => rng.GetNormalSingle(p1, p2);  // MathEx.CDFNormal(Random.Shared.NextSingle(), p1, p2);
        // henormal, sd = scale * sqrt(2 / fan)
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HeNormalKernel(SimpleRNG rng, float p1, float p2, float p3)
            => rng.GetNormalSingle(p1, p2 * MathF.Sqrt(2f / p3)); //  MathEx.CDFNormal(Random.Shared.NextSingle(), p1, p2) * p3;
        // henormal, sd = scale * sqrt(2 / (fanin+fanout))
        // p3 = fanin+fanout
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float GlorotKernel(SimpleRNG rng, float p1, float p2, float p3)
            => rng.GetNormalSingle(p1, p2 * MathF.Sqrt(6f / p3)); //  MathEx.CDFNormal(Random.Shared.NextSingle(), p1, p2) * p3;


        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float UniformKernel(float uniform, float lower, float upper, float p3)
            => lower + uniform * (upper - lower);
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LogNormalKernel(float uniform1, float uniform2, float p1, float p2, float p3)
        {
            // Use Box-Muller algorithm
            float u1 = uniform1;
            float u2 = uniform2;
            float r = MathF.Sqrt(-2F * MathF.Log(u1));
            float theta = 2F * MathF.PI * u2;
            r = r * MathF.Sin(theta);

            return MathF.Exp(p1 + p2 * r);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NormalKernel(float uniform1, float uniform2, float p1, float p2, float p3)
        {
            // Use Box-Muller algorithm
            float u1 = uniform1;
            float u2 = uniform2;
            float r = MathF.Sqrt(-2F * MathF.Log(MathF.Abs(u1)));
            float theta = 2F * MathF.PI * u2;
            r = r * MathF.Sin(theta);

            return p1 + p2 * r;
        }

        // henormal, sd = scale * sqrt(2 / fan)
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HeNormalKernel(float uniform1, float uniform2, float p1, float p2, float p3)
        {
            // henormal, sd = scale * sqrt(2 / (fanin+fanout))
            return NormalKernel(uniform1, uniform2, p1, p2 * MathF.Sqrt(2f / p3), 0);
        }

        // p3 = fanin+fanout
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float GlorotKernel(float uniform1, float uniform2, float p1, float p2, float p3)
          { 
            return NormalKernel(uniform1, uniform2, p1, p2 * MathF.Sqrt(6f / p3), 0); //  MathEx.CDFNormal(Random.Shared.NextSingle(), p1, p2) * p3;
        }


        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float XavierNormalizedKernel(float uniform1, float previous, float next)
        {
            float lower = 1 - (MathF.Sqrt(6f) / MathF.Sqrt(previous + next));
            float upper = (MathF.Sqrt(6f) / MathF.Sqrt(previous + next));
            return UniformKernel(uniform1, lower, upper, 0);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float XavierKernel(float uniform1, float previous)
        {
            float f = (1 / MathF.Sqrt(previous));
            return UniformKernel(uniform1, -f, f, 0);
        }


        public static RandomDistributionInfo Uniform(float low = 0, float high = 1)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.Uniform,
                RandomInputCount = 1,
                P1 = low,
                P2 = high,
                P3 = 0,
                Kernel = UniformKernel
            };
        }
        public static RandomDistributionInfo Normal(float mean, float sd)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.Normal,
                RandomInputCount = 2,
                P1 = mean,
                P2 = sd,
                P3 = 0,
                Kernel = NormalKernel
            };
        }
        public static RandomDistributionInfo LogNormal(float mean, float sd)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.LogNormal,
                RandomInputCount = 2,
                P1 = mean,
                P2 = sd,
                P3 = 0,
                Kernel = LogNormalKernel,
            };
        }
        public static RandomDistributionInfo HeNormal(float mean, float sd, float fan)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.HeNormal,
                RandomInputCount = 2,
                P1 = mean,
                P2 = sd,
                P3 = fan,
                Kernel = HeNormalKernel
            };
        }
        public static RandomDistributionInfo HeNormalFromSize(float mean, float sd, int distributionSize)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.HeNormal,
                RandomInputCount = 2,
                P1 = mean,
                P2 = sd,
                P3 = distributionSize,
                Kernel = HeNormalKernel
            };
        }
        public static float HeNormalFanSize(int distributionSize)
        {
            float fan = distributionSize * (float)Math.Sqrt(2f / distributionSize);
            return fan;
        }
        public static float GlorotFanSize(int totalFanSize)
        {
            float fan = totalFanSize * (float)Math.Sqrt(6f / totalFanSize);
            return fan;
        }

        public static RandomDistributionInfo GlorotFromSize(float mean, float scale, int sizeIn, int sizeOut)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.Glorot,
                RandomInputCount = 2,
                P1 = mean,
                P2 = scale,
                P3 = GlorotFanSize(sizeIn + sizeOut),
                Kernel = GlorotKernel
            };
        }

    }
}
