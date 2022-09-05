namespace NSS.Neural
{
    public struct Sample
    {
        public readonly int Class;
        public readonly int ClassCount;
        public readonly int Index;
        public readonly int Size;

        public float Variance;
        public float Average;
        public float Cost;
        public float CostChange;

        public readonly SampleEncodingType Encoding = SampleEncodingType.Normalized01;
        public readonly SampleExpectationEncodingType ExpectationEncoding = SampleExpectationEncodingType.OneHot;

        public Sample(int sampleIndex, int sampleSize, int _class, int classCount)
        {
            Index = sampleIndex;
            Size = sampleSize;
            Class = _class;
            ClassCount = classCount;
            Variance = 0;
            Average = 0;
            Cost = float.MaxValue;
            CostChange = 0;
        }
    }
}
