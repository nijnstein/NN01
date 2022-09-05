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

        // todo: Span2D<T>.Column => Column(int column); 
    }
}
