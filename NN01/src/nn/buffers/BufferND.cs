using ILGPU.IR.Values;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{

    public interface IBufferND<T> where T: unmanaged
    { 
        AlignedBuffer<T> Buffer { get; }

        void Lease(Action<IBufferLease<T>> action);

        bool IsLeased { get; }
    }

    public interface IBufferLease<T> where T : unmanaged
    {
        Span<T> Get(int bufferIndex);
        Memory<T> GetMemory(int bufferIndex);
        NetworkBuffers<T> GetLayerData(IBufferLease<T> lease, LayerInfo info);
        GPUNetworkBuffers<T> GetGPULayerData(IBufferLease<T> lease, LayerInfo info, Accelerator acc);
    }


    public abstract class Buffer256<T> : IDisposable, IBufferND<T>
        where T: unmanaged
    {
        const int Alignment = 32;  
        public AlignedBuffer<T> Buffer { get; private set; }
        public int Size { get; private set; }

        public Buffer256(int size, bool pooled = true)
        {
            Size = size;
            Buffer = new AlignedBuffer<T>(size, Alignment, pooled);
        }
        public Buffer256(int[] sizes, bool pooled = true)
        {
            int size = 0; 
            for(int i = 0; i < sizes.Length; i++)
            {
                size += sizes[i].Align256();
            }
            Size = size;
            Buffer = new AlignedBuffer<T>(Size, Alignment, pooled);
        }

        public virtual void Dispose()
        {
            if(Buffer != null)
            {
                Buffer.Dispose(); 
            }
        }

        public bool IsLeased
        {
            get
            {
                Debug.Assert(Buffer != null, "internal buffer == null");
                return Buffer.IsLeased; 
            }
        }

        public abstract void Lease(Action<IBufferLease<T>> action);
    }


    public struct BufferInfo256
    {
        public int BaseIndex { get; init; } 
        public int Index { get; init; }
        public int Length { get; init; }
        public int AlignedLength => Length.Align256(); 
    }

    // class buffer2d: buffernd<float> .... 

    public sealed class Buffers<T> : Buffer256<T>, IBufferLease<T> where T : unmanaged
    {
        public BufferInfo256[] BufferInfo { get; private set; }
        public int BufferCount => BufferInfo == null ? 0 : BufferInfo.Length;
        public long TotalLength => Buffer == null ? 0 : Buffer.Size;

        private AlignedBuffer<T>.Lease<T> CurrentLease = default; 

        public Buffers(int[] sizes, bool pooled = true) : base(sizes, pooled)
        {
            Debug.Assert(sizes != null); 
            BufferInfo = new BufferInfo256[sizes.Length];
            
            int index = 0;
            int layer = 0; 

            for(int i = 0; i < BufferInfo.Length; i++)
            {
                BufferInfo[i] = new BufferInfo256()
                {
                    BaseIndex = i,  
                    Index = index,
                    Length = sizes[i]
                };

                index += BufferInfo[i].AlignedLength;
            }            
        }

        public override void Dispose()
        {
            base.Dispose(); 
        }

        public override void Lease(Action<IBufferLease<T>> action)
        {
            Debug.Assert(Buffer != null, "internal buffer == null");
            Debug.Assert(!Buffer.IsLeased, "buffer already leased");

            Buffer.With((lease) =>
            {
                this.CurrentLease = lease;
                try
                {
                    action(this);
                }
                finally
                {
                    this.CurrentLease = default;
                }
            });
        }
        public Span<T> Get(int bufferIndex)
        {
            Debug.Assert(BufferInfo != null && Buffer != null, "internal buffer(s) not initialized");
            Debug.Assert(bufferIndex >= 0 && bufferIndex < BufferInfo.Length);
            Debug.Assert(Buffer.IsLeased);
            BufferInfo256 info = BufferInfo[bufferIndex];
            return CurrentLease.GetSpan(info.Index, info.Length); 
        }
        public Memory<T> GetMemory(int bufferIndex)
        {
            Debug.Assert(BufferInfo != null && Buffer != null, "internal buffer(s) not initialized");
            Debug.Assert(bufferIndex >= 0 && bufferIndex < BufferInfo.Length);
            Debug.Assert(Buffer.IsLeased);
            BufferInfo256 info = BufferInfo[bufferIndex];
            return CurrentLease.GetMemory(info.Index, info.Length);
        }
        public NetworkBuffers<T> GetLayerData(IBufferLease<T> lease, LayerInfo info)
        {
            Debug.Assert(Buffer.IsLeased, "buffer not leased");
            return new NetworkBuffers<T>(lease, info.LayerIndex, info.IsOutput);
        }
        public GPUNetworkBuffers<T> GetGPULayerData(IBufferLease<T> lease, LayerInfo info, Accelerator acc)
        {
            Debug.Assert(Buffer.IsLeased, "buffer not leased");
            Debug.Assert(acc != null, "GPU Accelerator null, cannot initialize layer data"); 
            return new GPUNetworkBuffers<T>(lease, info, acc);
        }

    }
}
