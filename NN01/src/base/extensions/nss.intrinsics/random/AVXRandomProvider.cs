using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    /// <summary>
    /// this should be faster then the gpu at the cost of using the avx ports
    /// - it depends on the usage scenario
    /// </summary>
    public class AVXRandomProvider : IRandomProvider
    {
        public bool GPUAccelerated => false;

        public RandomDistributionInfo DistributionInfo => throw new NotImplementedException();

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public void KickDirty(bool all = true, bool synchronizeAndCopy = true)
        {
            throw new NotImplementedException();
        }

        public RandomProviderLease Lease(bool waitOnDirty = true, CancellationToken cancelToken = default, int timeout = 100)
        {
            throw new NotImplementedException();
        }

        public bool LeaseDirtySpan(int length, out Span<float> span, bool waitForData = true, CancellationToken cancelToken = default, int timeout = 100)
        {
            throw new NotImplementedException();
        }

        public void Return(RandomProviderLease lease, bool kickIfNeeded = true)
        {
            throw new NotImplementedException();
        }
    }
}
