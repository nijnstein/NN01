using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public class CPURandom : IRandom
    {
        public RandomDistributionInfo Info { get; set; }

        public RandomDistributionType DistributionType => RandomDistributionType.Uniform;

        readonly float p1;
        readonly float p2; 

        public CPURandom(RandomDistributionInfo info)
        {
            Debug.Assert(info.DistributionType == RandomDistributionType.Uniform); 
            p1 = (Info.High - Info.Low);
            p2 = (Info.Low + Info.High) / 2; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Fill(Span<float> data) => Fill(data, 0, data.Length);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Fill(Span<float> data, int startIndex, int count)
        {
            unchecked
            {
                for (int i = startIndex, j = 0; j < count; i++, j++)
                {
                    data[i] = NextSingle();
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Next(int i)
        {
            return Random.Shared.Next(i); 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextSingle()
        {
            return p1 * Random.Shared.NextSingle() - p2; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<float> Span(int length)
        {
            Span<float> data = stackalloc float[length];
            unchecked
            {
                for (int i = 0; i < length; i++)
                {
                    data[i] = NextSingle();
                }
            }
            return data.ToArray().AsSpan();
        }
    }
}
