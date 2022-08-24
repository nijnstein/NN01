using NUnit.Framework;
using NN01;

namespace UnitTests
{
    [TestFixture]
    public class UnitTest
    {
        [TestCase(new float[] { 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5 }, 87)]
        [TestCase(new float[] { 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001 }, 1001)]
        public void Max(float[] a, float m)
        {
            float f = a.AsSpan().Max();
            Assert.IsTrue(f == m, $"{f} != {m}");
        }

    }
}
