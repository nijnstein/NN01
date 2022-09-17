using ILGPU.Runtime;
using NSS;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{
    public class Convolve2D : ConvolutionLayer
    {
        public override LayerInitializationType KernelInitializer => LayerInitializationType.Normal;

        public override LayerInitializationType BiasInitializer => LayerInitializationType.dot01;

        public override LayerType ActivationType => throw new NotImplementedException();

        public override LayerConnectedness Connectedness => LayerConnectedness.Convolutions;
        public Convolve2D(Size2D size, Size2D previous, Size2D kernel, bool skipInit = true, IRandom random = null)  
            : base(size, previous, kernel, skipInit, random) 
        { 
        
        }

        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            throw new NotImplementedException();
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            throw new NotImplementedException();
        }

        public override void Derivate(Span<float> input, Span<float> output)
        {
            throw new NotImplementedException();
        }
    }
}
