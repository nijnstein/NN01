namespace NSS
{
    public interface IRandom : IDisposable
    {
        RandomDistributionType DistributionType { get; }
        float NextSingle();
        int Next(int i);
        void Fill(Span<float> data);
        void Fill(Span<float> data, int startIndex, int count);
        Span<float> Span(int length);
    }
}
