using NUnit.Framework;
using NN01;
using System;
using BenchmarkDotNet.Attributes;
using System.Text;
using ILGPU.Algorithms.Random;
using NSS;
using NSS.GPU;

namespace UnitTests
{
    [TestFixture]
    public class UnitTest
    {

        [TestCase()]
        public void Span2D_1()
        {
            var data = new float[2, 2] { { 1, 2 }, { 3, 4 } };
            Span2D<float> s = data.AsSpan2D<float>();
            Assert.IsTrue(s.Column(0)[0] == 1f);
            Assert.IsTrue(s.Column(0)[1] == 3f);
            Assert.IsTrue(s.Column(1)[0] == 2f);
            Assert.IsTrue(s.Column(1)[1] == 4f);
            //Assert.IsTrue(s.Column(1).Sum() == 6f);
        }

        [TestCase(new float[] { 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5 }, 87)]
        [TestCase(new float[] { 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001 }, 1001)]
        public void Intrinsics_Max(float[] a, float m)
        {
            float f = Intrinsics.Max(a);
            Assert.IsTrue(f == m, $"intrinsics: {f} != {m}");
        }

        [TestCase(new float[] { 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5 }, 87)]
        [TestCase(new float[] { 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001 }, 1001)]
        public void MathEx_Max(float[] a, float m)
        {
            float f = MathEx.Max(a);
            Assert.IsTrue(f == m, $"mathex: {f} != {m}");
        }

        [TestCase(
            new float[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            new float[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            new float[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 })]
        [TestCase(
            new float[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
            new float[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
            new float[] { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 })]
        public void Intrinsics_Multiply(float[] a, float[] b, float[] c)
        {
            Intrinsics.Multiply(a, b, b);
            for (int i = 0; i < a.Length; i++)
            {
                Assert.IsTrue(b[i] == c[i]);
            }
        }

        [TestCase(new float[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, 36)]
        public void Intrinsics_Sum(float[] a, float b)
        {
            float f = Intrinsics.Sum(a);
            Assert.IsTrue(f == b, $"intrinsics {f} != {b}");
        }


        [TestCase(new float[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, 36)]
        public void MathEx_Sum(float[] a, float b)
        {
            float f = MathEx.Sum(a);
            Assert.IsTrue(f == b, $"mathex {f} != {b}");
        }

        [TestCase(new float[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 })]
        public void Intrinsics_Exp(float[] a)
        {
            float f = MathF.Exp(a[0]);
            Intrinsics.Exp(a);

            Assert.IsTrue(a[0] >= f - 0.00001f && a[0] <= f + 0.0001f, $"intrinsics.exp {a[0]} != {f}");
        }


        [TestCase]
        public void GPURandom1MxBlockSpan()
        {
            GPURandom rnd = new GPURandom(RandomDistributionInfo.Default); 

            for (int i = 0; i < 100000; i++)
            {
                Span<float> data = rnd.Span(256);

                Assert.IsTrue(data.Length > 0);
            }
        }


        [TestCase]
        public void GPURandom1MxBlockSpanNormal()
        {
            GPURandom rnd = new GPURandom(RandomDistributionInfo.Normal(0, 1));

            for (int i = 0; i < 100000; i++)
            {
                Span<float> data = rnd.Span(256);

                Assert.IsTrue(data.Length > 0);
            }
        }
        [TestCase]
        public void GPURandom1MxBlockSpanGaussian()
        {
            GPURandom rnd = new GPURandom(RandomDistributionInfo.LogNormal(0, 1));

            for (int i = 0; i < 100000; i++)
            {
                Span<float> data = rnd.Span(256);

                Assert.IsTrue(data.Length > 0);
            }
        }



        #region AlignedBuffer Tests 
        float[] fbuffer = new float[1000];

        [TestCase]
        public void AlignedBuffer_Bufferedx1000()
        {
            int i = 0;
            new AlignedBuffer<float>(fbuffer, 64, 32).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, 64);
                for (i = 0; i < 1000; i++)
                {
                    Intrinsics.Sum(s);
                }
            });
            Assert.IsTrue(i == 1000);
        }

        [TestCase]
        public void AlignedBuffer_PooledSingle()
        {
            bool b = false;
            new AlignedBuffer<float>(64, 32).With((lease) =>
            {
                Intrinsics.Sum(lease.GetSpan(0, 64));
                b = true;
            });
            Assert.IsTrue(b);
        }

        [TestCase]
        public void AlignedBuffer_Pooledx1000()
        {
            int i = 0;
            new AlignedBuffer<float>(64, 32).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, 64);
                for (i = 0; i < 1000; i++) Intrinsics.Sum(s);
            });
            Assert.IsTrue(i == 1000);
        }

        [TestCase]
        public void AlignedBuffer_HeapSingle()
        {
            bool b = false;
            new AlignedBuffer<float>(64, 32, false).With((lease) =>
            {
                Intrinsics.Sum(lease.GetSpan(0, 64));
                b = true;
            });
            Assert.IsTrue(b);
        }

        [TestCase]
        public void AlignedBuffer_Heapx1000()
        {
            int i = 0; 
            new AlignedBuffer<float>(64, 32, false).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, 64);
                for (i = 0; i < 1000; i++) Intrinsics.Sum(s);
            });
            Assert.IsTrue(i == 1000); 
        }
        #endregion 

    }
}
