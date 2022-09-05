using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NSS.GPU 
{
    /// <summary>
    /// provides single random numbers, can also fill blocks but with an additional buffer copy
    /// - to avoid the buffer copy use a span for acces, although the random numbers may be overwritten
    ///   by other random numbers from the same distribution 
    /// </summary>
    public class GPURandom : IRandom, IDisposable
    {
        public const int DefaultChunkCount = 4;
        public const int DefaultChunkSize = 1024 * 256;

        public RandomDistributionInfo Info => gpuProvider.DistributionInfo;
        public RandomDistributionType DistributionType => Info.DistributionType; 
        public bool IsGPUAccellerated => gpuProvider.GPUAccelerated;
        public int ChunkSize => ChunkSize;
        public int ChunkCount => ChunkCount; 

        int chunkSize;
        int chunkCount;
        bool ownProvider;
        GPURandomProvider gpuProvider;
        RandomProviderLease? lease;

        int cursor;
        float[] currentBuffer;

        public GPURandom(RandomDistributionInfo info) : this(info, DefaultChunkSize, DefaultChunkCount, null)
        {
        }

        public GPURandom(RandomDistributionInfo info, int chunkSize, int chunkCount, GPURandomProvider? provider, int seed = 0, bool init = true)
        {
            this.chunkSize = chunkSize;
            this.chunkCount = chunkCount;

            if (provider == null)
            {
                ownProvider = true;
                gpuProvider = new GPURandomProvider(info, chunkSize, chunkCount, seed == 0 ? (int)DateTime.Now.Ticks : seed);
            }
            else
            {
                ownProvider = false;
                gpuProvider = provider;
            }

            lease = null;
            cursor = -1;
            currentBuffer = null!;

            if (gpuProvider != null && init)
            {
                Task.Run(() => gpuProvider.KickDirty(all: true));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextSingle()
        {
            UpdateCurrentBuffer();
            unchecked
            {
                return currentBuffer[cursor++];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Next(int n)
        {
            return MathEx.FloorToInt(NextSingle() * n); 
        }


        public void Fill(Span<float> data, int startIndex, int count)
        {
            int n = count;
            int i = startIndex;

            while (n > 0)
            {
                UpdateCurrentBuffer();

                int c = Math.Min(n, chunkSize - cursor);
                currentBuffer.AsSpan().Slice(cursor, c).CopyTo(data.Slice(i));

                cursor += c;
                i = i + c;
                n = n - c;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateCurrentBuffer()
        {
            if (cursor < 0 | cursor >= chunkSize)
            {
                if (lease != null)
                {
                    gpuProvider.Return(lease);
                }
                lease = gpuProvider.Lease(true);
                cursor = lease!.chunk.dirtyCursor;
                currentBuffer = lease!.chunk.data;
            }
        }

        /// <summary>
        /// create a span over an internal buffer, this has the possibility of being overwritten with other numbers at a random 
        /// point in time (randoms from the same distribution) so keep care, depending on the usage scenario this might save the most
        /// expensive operation on the cpu: the block copy
        /// 
        /// - benchmarking reveals that the span-access is much faster as long as the gpu outputs numbers in time
        ///   when waiting on new chunks the span is only slightly faster as the copy operation does not take much time relative 
        ///   to generating the numbers on the gpu
        /// 
        /// </summary>
        /// <param name="length">the length that must be smaller then chunkszie </param>
        public Span<float> Span(int length)
        {
            Debug.Assert(length < chunkSize, "requested span may not span multiple internal buffers and must be shorter then blocksize");

            // make sure cursor is in a valid block 
            UpdateCurrentBuffer(); 

            // check if we can just use the current block 
            int n = length; 
            if(cursor + n < chunkSize)
            {
                Span<float> span = currentBuffer.AsSpan(cursor, length);
                cursor += length;
                return span; 
            }
            else
            {
                // take it from the first available block that is not leased out
                // that block will have its dirty cursor increased 
                // and potentially we have to wait for data from the gpu
                if (gpuProvider.LeaseDirtySpan(length, out Span<float> span, true))
                {
                    return span;
                }
                throw new Exception("all buffers in use");
            }
        }

        public void Fill(Span<float> data)
        {
            Fill(data, 0, data.Length);
        }

        public void Dispose()
        {
            if (ownProvider && gpuProvider != null)
            {
                gpuProvider.Dispose();
            }
        }
    }

}
