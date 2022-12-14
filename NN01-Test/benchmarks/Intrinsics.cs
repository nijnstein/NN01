using BenchmarkDotNet.Attributes;
using NN01;
using NSS;
using NUnit.Framework;
using System.Buffers;

namespace Test
{

    public class IntrinsicsBenchmark
    {
        static public float[] a = new float[]
        {
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001,
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001,
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001
        };


        [GlobalSetup]
        public void InitBuffer()
        {
        }

        // |    Method |      Mean |    Error |   StdDev |
        // |---------- |----------:|---------:|---------:|
        // |    MaxAVX |  12.95 ns | 0.116 ns | 0.103 ns |
        // |   MaxFast |  84.46 ns | 0.313 ns | 0.277 ns |
        // | MaxDotnet | 483.69 ns | 2.373 ns | 2.220 ns |

        [Benchmark] public float MaxAVX() => Intrinsics.Max(a);
        [Benchmark] public float MaxFast() => Intrinsics.MaxUnaligned(a);
        [Benchmark] public float MaxDotnet() => a.Max();


        //|             Method |         Mean |      Error |     StdDev |
        //|------------------- |-------------:|-----------:|-----------:|
        //|             SumAvx |     14.43 ns |   0.315 ns |   0.481 ns |
        //|            SumFast |     76.68 ns |   1.533 ns |   1.825 ns |
        //|          SumDotnet |    459.62 ns |   3.549 ns |   3.320 ns |
        //| SumAvxAlignedx1000 | 15,237.94 ns | 198.302 ns | 185.491 ns |         looks like there is no penalty for misalignment on this machine..
        [Benchmark] public float SumAvx() => Intrinsics.Sum(a);
        [Benchmark] public float SumFast() => Intrinsics.SumUnaligned(a);
        [Benchmark] public float SumDotnet() => a.Sum();     

        // | ExpAvxFast |  41.53 ns | 0.289 ns | 0.270 ns |
        // | Exp        | 453.79 ns | 3.111 ns | 2.758 ns |
        [Benchmark] public Span<float> ExpAvxFast() => MathEx.ExpFast(a);
        [Benchmark] public Span<float> Exp() => MathEx.Exp(a);
    }
}
