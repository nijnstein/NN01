using NSS.GPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS 
{
    public class RandomProviderChunk
    {
        public RandomProviderChunk(int index, float[] data)
        {
            this.index = index;
            this.data = data;
            leased = false;
            dirty = true;
            dirtyCursor = 0;
        }

        internal int index;
        internal float[] data;
        internal int dirtyCursor;
        internal bool dirty;
        internal bool leased;

        public Span<float> Data { get { return data.AsSpan(); } }
    }

    public class RandomProviderLease : IDisposable
    {
        internal IRandomProvider provider;
        internal RandomProviderChunk chunk;

        public RandomProviderLease(IRandomProvider provider, RandomProviderChunk chunk)
        {
            this.provider = provider;
            this.chunk = chunk;
        }

        public void Dispose()
        {
            provider.Return(this);
        }
    }
}
