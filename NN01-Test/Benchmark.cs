using BenchmarkDotNet.Attributes;
using NN01;

namespace UnitTests
{
    
    public class Benchmark
    {
        public float[] a = new float[]
        {
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001,
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001,
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001
        };

// |    Method |      Mean |    Error |   StdDev |
// |---------- |----------:|---------:|---------:|
// |    MaxAVX |  12.95 ns | 0.116 ns | 0.103 ns |
// |   MaxFast |  84.46 ns | 0.313 ns | 0.277 ns |
// | MaxDotnet | 483.69 ns | 2.373 ns | 2.220 ns |
            
//        [Benchmark] public float MaxAVX() => Intrinsics.Max(a);
//        [Benchmark] public float MaxFast() => Intrinsics.MaxUnaligned(a);
//        [Benchmark] public float MaxDotnet() => a.Max();


// |    Method |      Mean |    Error |   StdDev |
// |---------- |----------:|---------:|---------:|
// |    SumAvx |  14.43 ns | 0.179 ns | 0.167 ns |
// |   SumFast |  76.11 ns | 1.510 ns | 1.551 ns |
// | SumDotnet | 458.25 ns | 5.504 ns | 5.149 ns |
        [Benchmark] public float SumAvx() => Intrinsics.Sum(a);
        [Benchmark] public float SumFast() => Intrinsics.SumUnaligned(a);
        [Benchmark] public float SumDotnet() => a.Sum();

    }
}
