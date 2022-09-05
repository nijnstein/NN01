using ILGPU.Algorithms.Random;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

        public static RandomDistributionInfo Uniform(float low, float high)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.Uniform,
                RandomInputCount = 1,
                P1 = low,
                P2 = high,
                P3 = 0,
            };
        }
        public static RandomDistributionInfo Normal(float mean, float sd)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.Normal,
                RandomInputCount = 1,
                P1 = mean,
                P2 = sd,
                P3 = 0,
            };
        }
        public static RandomDistributionInfo Gaussian(float mean, float sd)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.Gaussian,
                RandomInputCount = 2,
                P1 = mean,
                P2 = sd,
                P3 = 0,
            };
        }
        public static RandomDistributionInfo HeNormal(float mean, float sd, float fan)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.HeNormal,
                RandomInputCount = 1,
                P1 = mean,
                P2 = sd,
                P3 = fan,
            };
        }
        public static RandomDistributionInfo HeNormalFromSize(float mean, float sd, int distributionSize)
        {
            return new RandomDistributionInfo()
            {
                ProviderBackend = DefaultProviderBackend,
                DistributionType = RandomDistributionType.HeNormal,
                RandomInputCount = 1,
                P1 = mean,
                P2 = sd,
                P3 = HeNormalFanSize(distributionSize),
            };
        }
        public static float HeNormalFanSize(int distributionSize)
        {
            float fan = distributionSize * (float)Math.Sqrt(1f / distributionSize);
            return fan;
        }
    }
}
