using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static ILGPU.IR.Analyses.Uniforms;

namespace NSS
{
    public interface IRandom
    {
        RandomDistributionType DistributionType { get; }
        float NextSingle();
        int Next(int i);
        void Fill(Span<float> data);
        void Fill(Span<float> data, int startIndex, int count);
        Span<float> Span(int length);
    }
}
