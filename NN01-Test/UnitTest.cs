using NUnit.Framework;
using NN01;

namespace UnitTests
{
    [TestFixture]
    public class UnitTest
    {
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
            for(int i = 0; i < a.Length; i++)
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
    }
}
