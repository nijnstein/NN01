namespace NSS
{
    public sealed class GaussianRandom : CPURandom
    {
        public readonly float Mean;
        public readonly float SD;

        public GaussianRandom(float mean, float sd) : base(RandomDistributionInfo.Normal(mean, sd))
        {
            Mean = mean;
            SD = sd; 
        }
    }
}
