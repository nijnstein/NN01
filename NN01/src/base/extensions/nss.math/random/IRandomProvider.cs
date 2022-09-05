namespace NSS 
{
    public interface IRandomProvider: IDisposable
    {   
        bool GPUAccelerated { get; }
        RandomDistributionInfo DistributionInfo { get; }
        void KickDirty(bool all = true, bool synchronizeAndCopy = true);
        RandomProviderLease Lease(bool waitOnDirty = true, CancellationToken cancelToken = default, int timeout = 100);
        bool LeaseDirtySpan(int length, out Span<float> span, bool waitForData = true, CancellationToken cancelToken = default, int timeout = 100);
        void Return(RandomProviderLease lease, bool kickIfNeeded = true);
    }
}