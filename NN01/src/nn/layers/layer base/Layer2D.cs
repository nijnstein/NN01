using NSS;

namespace NN01
{
    public abstract class Layer2D : Layer
    {
        public readonly Size2D Size2D; 
        public int Width => Size2D.X;
        public int Height => Size2D.Y;

        public Span2D<float> Neurons2D => Neurons.AsSpan2D<float>();

        public Layer2D(Size2D size, Size2D previousSize) : base(size.X * size.Y, previousSize.X * previousSize.Y)
        {
            this.Size2D = size; 
        }   
    }
}
