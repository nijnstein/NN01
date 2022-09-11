using BenchmarkDotNet.Attributes;
using Iced.Intel;
using NN01;
using NSS;
using NSS.GPU;
using NUnit.Framework;
using System.Buffers;
using System.Runtime.InteropServices;

namespace Test
{

    public class GPURandomBenchmarks
    {

        private GPURandom rnd;
        private GPURandom rndgauss;
        private GPURandom rndnormal;
        private Random cpurnd;
        private float[] block;

        const int BlockSize = 1024 * 16;

        [GlobalSetup]
        public void InitBuffer()
        {
            rnd = new GPURandom(RandomDistributionInfo.Uniform(0, 1), 128 * 2048, 3, null);
            rndgauss = new GPURandom(RandomDistributionInfo.LogNormal(0, 1), 128 * 2048, 3, null);
            rndnormal = new GPURandom(RandomDistributionInfo.LogNormal(0, 1), 128 * 2048, 3, null);
            cpurnd = new Random();
            block = new float[BlockSize];
        }


//           
//                   Intel Core i9-9900K CPU 3.60GHz(Coffee Lake), 1 CPU, 16 logical and 8 physical cores
//                   .NET SDK= 7.0.100-preview.5.22307.18
//           
//                     [Host]     : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT
//             DefaultJob : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT
//           
//           
//           |                     Method |            Mean |         Error |         StdDev |
//           |--------------------------- |----------------:|--------------:|---------------:|
//           |            GPURandomSingle |       0.8779 ns |     0.0317 ns |      0.0281 ns |
//           |      GPURandomSingleNormal |       2.0819 ns |     0.1471 ns |      0.4221 ns |
//           |    GPURandomSingleGaussian |       2.1633 ns |     0.0613 ns |      0.0512 ns |
//           |             GPURandomBlock |  29,205.8866 ns |   889.3883 ns |  2,622.3813 ns |
//           |       GPURandomBlockNormal |  66,611.6395 ns | 5,082.4104 ns | 14,985.6018 ns |
//           |     GPURandomBlockGaussian |  70,215.1190 ns | 1,427.1092 ns |  4,071.6227 ns |
//           |         GPURandomBlockSpan |  20,267.8236 ns |   402.0929 ns |    463.0508 ns |
//           |   GPURandomBlockSpanNormal |  68,566.5396 ns | 1,918.3620 ns |  5,251.4849 ns |
//           | GPURandomBlockSpanGaussian |  44,202.2388 ns |   874.8503 ns |    818.3355 ns |
//           |            CPURandomSingle |       5.0382 ns |     0.0495 ns |      0.0463 ns |
//           |            CPURandomNormal |       3.8303 ns |     0.0764 ns |      0.0678 ns |
//           |          CPURandomGaussian |       9.1959 ns |     0.2142 ns |      0.2708 ns |
//           |             CPURandomBlock |  72,786.7864 ns |   370.3067 ns |    309.2229 ns |
//           |       CPURandomBlockNormal | 215,748.0530 ns | 4,141.4350 ns |  4,067.4429 ns |
//           |     CPURandomBlockGaussian | 468,632.7148 ns | 3,593.4157 ns |  3,361.2832 ns |
//           
        [Benchmark]
        public void GPURandomSingle()
        {
            rnd.NextSingle();
        }

        [Benchmark]
        public void GPURandomSingleNormal()
        {
            rndnormal.NextSingle();
        }

        [Benchmark]
        public void GPURandomSingleGaussian()
        {
            rndgauss.NextSingle();
        }

        [Benchmark]
        public void GPURandomBlock()
        {
            rnd.Fill(block);
        }

        [Benchmark]
        public void GPURandomBlockNormal()
        {
            rndnormal.Fill(block);
        }

        [Benchmark]
        public void GPURandomBlockGaussian()
        {
            rndgauss.Fill(block);
        }

        [Benchmark]
        public void GPURandomBlockSpan()
        {
            Span<float> data = rnd.Span(BlockSize);
        }

        [Benchmark]
        public void GPURandomBlockSpanNormal()
        {
            Span<float> data = rndnormal.Span(BlockSize);
        }

        [Benchmark]
        public void GPURandomBlockSpanGaussian()
        {
            Span<float> data = rndgauss.Span(BlockSize);
        }

        [Benchmark]
        public void CPURandomSingle()
        {
            float f = cpurnd.NextSingle();
        }

        [Benchmark]
        public void CPURandomNormal()
        {
            float f = MathEx.Normal(cpurnd, 0, 1);
        }

        [Benchmark]
        public void CPURandomGaussian()
        {
            float f = MathEx.Gaussian(cpurnd, 0, 1);
        }


        [Benchmark]
        public void CPURandomBlock()
        {
            unchecked
            {
                for (int j = 0; j < BlockSize; j++)
                {
                    block[j] = cpurnd.NextSingle() * 5f;
                }
            }
        }

        [Benchmark]
        public void CPURandomBlockNormal()
        {
            unchecked
            {
                for (int j = 0; j < BlockSize; j++)
                {
                    block[j] = MathEx.Normal(cpurnd, 0, 1);
                }
            }
        }

        [Benchmark]
        public void CPURandomBlockGaussian()
        {
            unchecked
            {
                for (int j = 0; j < BlockSize; j++)
                {
                    block[j] = MathEx.Gaussian(cpurnd, 0, 1);
                }
            }
        }
    }
}
