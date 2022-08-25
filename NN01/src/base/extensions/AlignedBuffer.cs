using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Diagnostics.SymbolStore;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    /// <summary>
    /// an array, aligned for SSE/AVX operations.
    ///
    /// - NOT ENTIRELY SAFE - 
    /// 
    /// </summary>
    public sealed class AlignedBuffer<T> : IDisposable where T: unmanaged 
    {
        public static readonly Action<object, Action<IntPtr>> GetPinnedPtr;

        /// <summary>
        /// magic: get addres from pinned pointer without going unsafe
        /// </summary>
        static AlignedBuffer()
        {
            var dyn = new DynamicMethod("GetPinnedPtr", typeof(void), new[] { typeof(object), typeof(Action<IntPtr>) }, typeof(AlignedBuffer<T>).Module);
            var il = dyn.GetILGenerator();
            il.DeclareLocal(typeof(object), true);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc_0);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Call, typeof(Action<IntPtr>).GetMethod("Invoke"));
            il.Emit(OpCodes.Ret);
            GetPinnedPtr = (Action<object, Action<IntPtr>>)dyn.CreateDelegate(typeof(Action<object, Action<IntPtr>>));
        }

        /// <summary>
        /// access to any slot outside [_base, _base + _size) is prohibited and asserted on
        /// </summary>
        private T[] items;

        /// <summary>
        /// Must be divisible by (_cbAlign / sizeof(Float)).
        /// </summary>
        public int Size { get; private set; }

        /// <summary>
        /// The alignment in bytes, a power of two, divisible by sizeof(Float)
        /// </summary>
        public int Alignment { get; private set; }

        public bool IsLeased { get; private set; }

        /// <summary>
        /// bese index of array from which stuff is aligned 
        /// </summary>
        private int _base;
        private int _sizeOfT;

        /// <summary>
        /// size of elements in bytes (sizeof(T))
        /// </summary>
        public int ElementSize => _sizeOfT; 
        public bool IsPooled { get; private set; }
        public bool IsOwned { get; private set; }

        /// <summary>
        /// Allocate an aligned vector with the given alignment (in bytes) in powers of 2 larger then 4
        /// </summary>
        public AlignedBuffer(int size, int alignment, bool pooled = true)
        {
            T t = default;
            _sizeOfT = Marshal.SizeOf<T>(t); 
            Debug.Assert(0 < size);
            Debug.Assert(_sizeOfT <= alignment);
            Debug.Assert((alignment & (alignment - 1)) == 0);
            Debug.Assert((size * _sizeOfT) % alignment == 0);

            if (pooled)
            {
                items = ArrayPool<T>.Shared.Rent(size + alignment / _sizeOfT);
            }
            else
            {
                items = new T[size + alignment / _sizeOfT]; 
            }

            Size = size;
            Alignment = alignment;
            IsLeased = false;
            IsPooled = pooled;
            IsOwned = true;

            _base = 0;
        }

        private T refT; 

        public AlignedBuffer(T[] data, int size, int alignment)
        {
            T t = default;
            _sizeOfT = Marshal.SizeOf<T>(t);
 
            Debug.Assert(0 < size);
            Debug.Assert(_sizeOfT <= alignment);
            Debug.Assert((alignment & (alignment - 1)) == 0);
            Debug.Assert((size * _sizeOfT) % alignment == 0);
            Debug.Assert(size + alignment / _sizeOfT <= data.Length);

            items = data; 
            Size = size;
            Alignment = alignment;
            IsLeased = false;
            IsPooled = false;
            IsOwned = false;

            _base = 0;
        }

        public void Dispose()
        {
            if (IsOwned)
            {
                Debug.Assert(items != null);
                if (IsPooled)
                {
                    ArrayPool<T>.Shared.Return(items);
                }
            }
        }

        private int GetBase(long addr)
        {
            Debug.Assert(IsLeased);

#if DEBUG
//            unsafe fixed (float* p = Items)
//            {
//                Debug.Assert((float*)addr == p);
//            }
#endif
            int low = (int)(addr & (Alignment - 1));
            int min = low == 0 ? 0 : Alignment - low;

            Debug.Assert(min % _sizeOfT == 0);

            int index = min / _sizeOfT;
            if (index == _base)
            {
                return _base;
            }

            _base = index;
            return _base;
        }

        public struct Lease<T> where T : unmanaged
        {
            private T[] items;
            private int @base;
            private int length;
            public Lease(T[] items, int @base, int length)
            {
                this.items = items;
                this.@base = @base;
                this.length = length;
            }

            /// <summary>
            /// get a full span over the buffer, the buffer may be garbage initialized
            /// </summary>
            public Span<T> AsSpan() => items.AsSpan(@base, length);

            /// <summary>
            /// get a span over the buffer, the buffer may be garbage initialized
            /// </summary>
            public Span<T> GetSpan(int from, int _length) => items.AsSpan(@base + from, _length);
        }

        /// <summary>
        /// lease the aligned buffer and pin its reference so the gc wont move it during execution 
        /// of the reference 
        /// </summary>
        /// <param name="action">action to perform on the lease</param>
        public void With(Action<Lease<T>> action)
        {
            if (IsOwned)
            {
                GetPinnedPtr(
                    items,
                    (Action<IntPtr>)((ptr) =>
                        {
                            Debug.Assert(!IsLeased, "AlignedBuffer already in use");
                            IsLeased = true;
                            try
                            {
                                action(new Lease<T>(items, GetBase(ptr.ToInt64()), Size));
                            }
                            finally
                            {
                                IsLeased = false;
                            }
                        })
                );
            }
        }
    }
}
