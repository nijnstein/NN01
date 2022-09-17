using System.Reflection.Metadata.Ecma335;

namespace NSS
{
    public sealed class UniformRandom : CPURandom
    {
        public readonly float Low;
        public readonly float High;

        public UniformRandom(float low, float high) : base(RandomDistributionInfo.Uniform(low, high))
        {
            Low = low;
            High = high; 
        }
    }
}
