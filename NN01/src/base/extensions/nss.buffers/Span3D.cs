using System.Runtime.CompilerServices;

namespace NSS
{
    public ref struct Span3D<T> where T : struct
    {
        public Span<T> Span;
        public readonly int X;
        public readonly int Y;
        public readonly int Z;
        private readonly int YZ; 

        public Span3D(Span<T> span, int x, int y, int z)
        {
            X = x;
            Y = y;
            Z = z;
            YZ = y * z; 
            Span = span;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetIndex(int x, int y, int z) => x * YZ + y * Z + z;

        public T this[int x, int y, int z]
        {
            get
            {
                return Span[GetIndex(x, y, z)];
            }
            set
            {
                Span[GetIndex(x, y, z)] = value;
            }
        }
        public Span<T> Slice(int x, int y, int length) => Span.Slice(GetIndex(x, y, 0), length);
        public Span<T> Row(int x, int y) => Span.Slice(GetIndex(x, y, 0), Z);
        public ColumnSpan Column(int x, int z) => new Span3D<T>.ColumnSpan(Span, YZ, GetIndex(x, 0, z));
        public Span2D<T> Span2D(int x) => new Span2D<T>(Span.Slice(GetIndex(x, 0, 0), YZ), Y, Z); 

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
