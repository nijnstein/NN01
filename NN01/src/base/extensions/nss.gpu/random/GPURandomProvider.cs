using ILGPU;
using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;

namespace NSS.GPU
{

    /// <summary>
    /// A gpu backed random number generator 
    /// </summary>
    public class GPURandomProvider : IDisposable, IRandomProvider
    {
        readonly object locker;
        public bool GPUAccelerated { get; protected set; }
        public RandomDistributionInfo DistributionInfo { get; protected set; }

        GPUContext? gpuContext;
        Accelerator? accelerator;
        MemoryBuffer1D<float, Stride1D.Dense> gpuMemory;

        RandomProviderChunk[] chunks;
        readonly int chunkSize;
        readonly int chunkCount;

        int lastLeaseIndex;
        int currentSeed;
        bool processingQueue;

        Queue<RandomProviderChunk> dirtyQueue;
        Task currentTask;                      

        public GPURandomProvider(RandomDistributionInfo info, int _chunkSize, int _chunkCount, int seed)
        {
            Debug.Assert(_chunkSize > 0, "chunk size must be larger then 0, much larger");
            Debug.Assert(_chunkCount > 1, "chunk count must be larger then 1");

            DistributionInfo = info;

            locker = new object();
            chunkSize = _chunkSize;
            chunkCount = _chunkCount;
            lastLeaseIndex = 0;
            currentSeed = seed;

            dirtyQueue = new Queue<RandomProviderChunk>();
            chunks = new RandomProviderChunk[chunkCount];

            for (int i = 0; i < chunkCount; i++)
            {
                chunks[i] = new RandomProviderChunk(i, new float[chunkSize]);
                dirtyQueue.Enqueue(chunks[i]);
            }

            if (GPUContext.HaveGPUAcceleration)
            {
                gpuContext = GPUContext.Create();
                accelerator = gpuContext.CreateGPUAccelerator();
                // some distribution kernels need multiple random inputs, so the actually allocated size of the buffers may be a multiple of chunksize 
                gpuMemory = accelerator.Allocate1D<float>((long)chunkSize * info.RandomInputCount);
                GPUAccelerated = true;
            }
            else
            {
                gpuContext = GPUContext.Create();
                accelerator = gpuContext.CreateCPUAccelerator();
                gpuMemory = accelerator.Allocate1D<float>((long)chunkSize * info.RandomInputCount);
                GPUAccelerated = false;
            }

            processingQueue = false;
            currentTask = null;
        }

        public RandomProviderLease Lease(bool waitOnDirty = true, CancellationToken cancelToken = default, int timeout = 100)
        {
            RandomProviderLease lease = DoLease();
            if (lease == null)
            {
                if (dirtyQueue.Count > 0)
                {
                    do
                    {
                        if (!processingQueue || (currentTask == null || currentTask.IsCompleted || currentTask.IsCanceled))
                        {
                            processingQueue = true;
                            currentTask = Task.Run(() => KickDirty(false));
                        }

                        if (waitOnDirty)
                        {
                            currentTask.Wait(timeout, cancelToken);
                            lease = DoLease();
                        }
                    }
                    while (waitOnDirty && lease == null && dirtyQueue.Count > 0);
                }
                else
                {
                    // no lease available
                    // - dont wait on a lease becoming free, we might deadlock easily depending on use
                }
            }
            return lease!;
        }

        RandomProviderLease DoLease()
        {
            int c = 0;
            int i = lastLeaseIndex + 1;

            lock (locker)
            {
                do
                {
                    if (i >= chunkCount) i = 0;

                    if (!chunks[i].dirty && !chunks[i].leased)
                    {
                        chunks[i].leased = true;
                        break;
                    }
                    else
                    {
                        c++;
                        i++;
                    }
                }
                while (c < chunkCount);
            }
            if (c == chunkCount)
            {
                return null!;
            }
            else
            {
                lastLeaseIndex = i;
                return new RandomProviderLease(this, chunks[i]);
            }
        }

        public void Return(RandomProviderLease lease, bool kickIfNeeded = true)
        {
            lock (locker)
            {
                lease.chunk.dirty = true;
                lease.chunk.dirtyCursor = 0;
                lease.chunk.leased = false;

                dirtyQueue.Enqueue(lease.chunk);
            }
            if (!processingQueue & kickIfNeeded)
            {
                currentTask = Task.Run(() => KickDirty(false));
            }
        }

        public void KickDirty(bool all = true, bool synchronizeAndCopy = true)
        {
            processingQueue = true;
            try
            {
                if (accelerator == null)
                {
                    throw new ArgumentNullException("accelerator");
                }
                while (dirtyQueue.Count > 0)
                {
                    lock (locker)
                    {
                        if (dirtyQueue.Count > 0)
                        {
                            RandomProviderChunk dirty = dirtyQueue.Dequeue();

                            switch (DistributionInfo.ProviderBackend)
                            {
                                case RandomProviderBackend.XorShift128Plus:
                                    {
                                        using (var rng1 = RNG.Create<XorShift128Plus>(accelerator, new Random(currentSeed)))
                                        {
                                            rng1.FillUniform(gpuMemory.View);
                                        }
                                    }
                                    break;

                                case RandomProviderBackend.XorShift128:
                                    {
                                        using (var rng1 = RNG.Create<XorShift128>(accelerator, new Random(currentSeed)))
                                        {
                                            rng1.FillUniform(gpuMemory.View);
                                        }
                                    }
                                    break;

                                case RandomProviderBackend.XorShift64star:
                                    {
                                        using (var rng1 = RNG.Create<XorShift64Star>(accelerator, new Random(currentSeed)))
                                        {
                                            rng1.FillUniform(gpuMemory.View);
                                        }
                                    }
                                    break;

                                case RandomProviderBackend.XorShift32:
                                    {
                                        using (var rng1 = RNG.Create<XorShift32>(accelerator, new Random(currentSeed)))
                                        {
                                            rng1.FillUniform(gpuMemory.View);
                                        }
                                    }
                                    break;
                            }

                            switch (DistributionInfo.DistributionType)
                            {
                                case RandomDistributionType.Uniform:
                                    {
                                        // we may need to scale the default uniform 
                                        if (DistributionInfo.P1 != 0f | DistributionInfo.P2 != 1f)
                                        {
                                            float mean = (DistributionInfo.P1 + DistributionInfo.P2) / 2;
                                            float scale = (DistributionInfo.P2 - DistributionInfo.P1) / 2;
                                            accelerator!.LaunchAutoGrouped<Index1D, ArrayView<float>, float, float>(
                                                UniformKernel, chunkSize, gpuMemory.View, scale, mean);
                                        }
                                    }
                                    break;

                                case RandomDistributionType.Gaussian:
                                    {
                                        accelerator!.LaunchAutoGrouped<Index1D, ArrayView<float>, float, float>(
                                                GaussianGPUKernel, chunkSize, gpuMemory.View, DistributionInfo.P1, DistributionInfo.P2);
                                    }
                                    break;

                                case RandomDistributionType.Normal:
                                    {
                                        accelerator!.LaunchAutoGrouped<Index1D, ArrayView<float>, float, float>(
                                                NormalGPUKernel, chunkSize, gpuMemory.View, DistributionInfo.P1, DistributionInfo.P2);
                                    }
                                    break;

                                case RandomDistributionType.HeNormal:
                                    {
                                        accelerator!.LaunchAutoGrouped<Index1D, ArrayView<float>, float, float, float>(
                                                HeNormalGPUKernel, chunkSize, gpuMemory.View, DistributionInfo.P1, DistributionInfo.P2, DistributionInfo.P3);
                                    }
                                    break;

                                default: throw new NotImplementedException();
                            }

                            if (synchronizeAndCopy)
                            {
                                // sync any running task on the gpu 
                                accelerator.Synchronize();

                                // copy the random numbers from gpu to our cpu data
                                // TODO -> a random class only supporting blocks of random numbers could eliminate this copy
                                if (DistributionInfo.RandomInputCount > 1)
                                {
                                    gpuMemory.View.SubView(0, chunkSize).CopyToCPU(accelerator!.DefaultStream, dirty.data);
                                }
                                else
                                {
                                    gpuMemory.View.CopyToCPU(accelerator!.DefaultStream, dirty.data);
                                }
                            }

                            // unflag dirty and rebase the current seed 
                            dirty.dirty = false;
                            dirty.dirtyCursor = 0;
                            currentSeed = (int)(dirty.data[0] * 10000000f);
                        }
                    }
                    if (!all)
                    {
                        return;
                    }
                }
            }
            finally
            {
                processingQueue = false;
                currentTask = null!;
            }
        }

        public bool LeaseDirtySpan(int length, out Span<float> span, bool waitForData = true, CancellationToken cancelToken = default, int timeout = 100)
        {
            if (length > chunkSize)
            {
                span = Span<float>.Empty;
                return false;
            }
            Task task = currentTask;

            bool all_leased = true;
            bool enqueued_dirty = false;
            bool have_dirty = false;

            for (int i = 0; i < chunks.Length; i++)
            {
                RandomProviderChunk c = chunks[i];

                if ((!c.dirty) & (!c.leased))
                {
                    if (length <= (chunkSize - c.dirtyCursor))
                    {
                        // take the span 
                        span = c.data.AsSpan(c.dirtyCursor, length);

                        // mark as dirty 
                        c.dirtyCursor += length;
                        return true;
                    }
                    else
                    {
                        enqueued_dirty = true;
                        c.dirty = true;
                        dirtyQueue.Enqueue(c);
                    }
                }

                have_dirty = have_dirty | c.dirty;
            }

            if (enqueued_dirty)
            {
                processingQueue = true;
                task = Task.Run(() => KickDirty(true));
                currentTask = task;
                if (waitForData)
                {
                    task.Wait(timeout, cancelToken);
                    return LeaseDirtySpan(length, out span, waitForData, cancelToken, timeout);
                }
            }
            else
            if (have_dirty && waitForData)
            {
                if (currentTask != null && !(currentTask.IsCompleted || currentTask.IsCanceled))
                {
                    currentTask.Wait(timeout, cancelToken);
                }
                else
                {
                    // most likely another thread is processing chunks, kick the schedular then retry to lease
                    Thread.Sleep(0);
                }
                return LeaseDirtySpan(length, out span, waitForData, cancelToken, timeout);
            }

            // all leases taken, or dirty and not waiting
            span = Span<float>.Empty;
            return false;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void GaussianGPUKernel(Index1D index, ArrayView<float> data, float mean, float sd)
        {
            // data.length should be 2x larger then index 
            float x1 = 1f - data[index];
            float x2 = 1f - data[data.Length / 2 + index];

            float y1 = MathF.Sqrt(-2f * MathF.Log(x1)) * MathF.Cos(2f * MathF.PI * x2);
            data[index] = y1 * sd + mean;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void UniformKernel(Index1D index, ArrayView<float> data, float scale, float mean)
        {
            data[index] = data[index] * scale + mean;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void NormalGPUKernel(Index1D index, ArrayView<float> data, float mean, float sd)
        {
            float x = data[index];
            float p = 1f / MathF.Sqrt(2f * MathF.PI * (sd * sd));
            data[index] = p * MathF.Exp(-0.5f * ((x - mean) * (x - mean)) / (sd * sd));
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void HeNormalGPUKernel(Index1D index, ArrayView<float> data, float mean, float sd, float fan)
        {
            float x = data[index];
            float p = 1f / MathF.Sqrt(2f * MathF.PI * (sd * sd));
            float n = p * MathF.Exp(-0.5f * ((x - mean) * (x - mean)) / (sd * sd));
            data[index] = fan * n;
        }

        public void Dispose()
        {
            if (gpuMemory != null) gpuMemory.Dispose();
            if (accelerator != null) accelerator.Dispose();
            if (gpuContext != null) gpuContext.Dispose();
        }
    }

}
