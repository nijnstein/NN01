using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NSS
{
    public sealed class BernoulliRandom : CPURandom
    {
        /// <summary>
        /// probability of 1 in output 
        /// </summary>
        public readonly float Factor;

        public BernoulliRandom(float factor) : base(RandomDistributionInfo.Uniform(0, 1f))
        {
            Factor = factor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override float NextSingle() => base.NextSingle() < Factor ? 1f : 0f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool NextBoolean() => base.NextSingle() < Factor;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Fill(BitBuffer buffer)
        {
            unchecked
            {
                for (int i = 0; i < buffer.BitCount; i++)
                {
                    buffer.Set(i, base.NextSingle() < Factor ? true : false);
                }
            }
        }
    }
}
