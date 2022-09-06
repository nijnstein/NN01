using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    /// <summary>
    /// Span over a 2 dimensional array 
    /// </summary>
    public ref struct Span2D<T> where T : struct
    {
        public Span<T> Span;
        public readonly int Width;
        public readonly int Height;

        public Span2D(Span<T> span, int rowCount, int columnCount)
        {
            Width = columnCount;
            Height = rowCount;
            Span = span;
        }

        public T this[int row, int column]
        {
            get
            {
                return Span[row * Width + column];
            }
            set
            {
                Span[row * Width + column] = value;
            }
        }
        public Span<T> Slice(int index, int length) => Span.Slice(index, length); 
        public Span<T> Row(int row) => Span.Slice(row * Width, Width);
        public ColumnSpan Column(int column) => new Span2D<T>.ColumnSpan(Span, Width, column);

        public ref struct ColumnSpan
        {
            public Span<T> Span;
            public int ColumnIndex;
            public int Stride;

            public int ElementCount => Span.Length / Stride;

            public ColumnSpan(Span<T> span, int stride, int column)
            {
                Span = span;
                Stride = stride;
                ColumnIndex = column; 
            }

            public T this[int row]
            {
                get
                {
                    return Span[row * Stride + ColumnIndex];
                }
                set
                {
                    Span[row * Stride + ColumnIndex] = value;
                }
            }

            public T[] ToArray()
            {
                T[] a = new T[ElementCount];
                CopyTo(a.AsSpan());
                return a; 
            }
            
            public void CopyTo(Span<T> other) => CopyTo(other, 0, Span.Length / Stride);
            
            public void CopyTo(Span<T> other, int from, int length)
            {
                for (int j = 0, i = from; i < from + length; i++, j++)
                {
                    other[j] = this[i];
                }
            }
        }
    }
}
