using NSS;

namespace NN01
{

    public abstract class ConvolutionLayer : Layer2D
    {
        public readonly Size2D KernelSize;


        // Padding: if enabled we add 0's on each border to strengthen the response 
        // on corner elements as the convolution will not give strong presence to corner data
        public readonly Size2D PaddingSize; 

        public float[,] Kernel;
        public float Bias;
        public override bool HasParameters => true;

        protected ConvolutionLayer(Size2D size, Size2D previousSize, Size2D kernelSize, bool skipInit = true, IRandom random = null)
            :this(
                 size,
                 previousSize, 
                 kernelSize, 
                 // default padding to make output the same size as input 
                 new Size2D(kernelSize.X - 1, kernelSize.Y - 1),
                 skipInit, 
                 random)
        {
        }

        protected ConvolutionLayer(Size2D size, Size2D previousSize, Size2D kernelSize, Size2D padding, bool skipInit = true, IRandom random = null)
            : base(
                  // size = size + padding in 2D
                  new Size2D(size.X + padding.X, size.Y + padding.Y), 
                  // previous size is what we are given
                  previousSize)
        {
            KernelSize = kernelSize;
            PaddingSize = padding;
            Kernel = new float[KernelSize.Y, KernelSize.X]; 

            if (!skipInit)
            {
                InitializeParameters(random);
            }
        }

        public abstract LayerInitializationType KernelInitializer { get; }
        public abstract LayerInitializationType BiasInitializer { get; }

        public void InitializeParameters(IRandom random)
        {
            bool ownRandom = random == null;
            if (ownRandom)
            {
                random = new CPURandom(RandomDistributionInfo.Uniform(0, 1f));
            }

            Bias = 0.1f;
            FillParameterDistribution(KernelInitializer, Kernel.AsSpan2D<float>().Span, random!);

            if (ownRandom)
            {
                random!.Dispose();
            }
        }
    }
}
