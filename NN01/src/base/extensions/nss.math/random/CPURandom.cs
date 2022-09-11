using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Printing;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{


    public class CPURandom : IRandom
    {
        private SimpleRNG rng;
        public RandomDistributionInfo Info { get; set; }

        public RandomDistributionType DistributionType => RandomDistributionType.Uniform;

        public CPURandom(RandomDistributionInfo info)
        {
            //Debug.Assert(info.DistributionType == RandomDistributionType.Uniform);
            rng = new SimpleRNG();
            // rng.SetSeedFromSystemTime(DateTime.Now);
            Info = info;
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
            return rng.GetInt(i); 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextSingle()
        {
            return Info.Kernel(rng, Info.P1, Info.P2, Info.P3);
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
