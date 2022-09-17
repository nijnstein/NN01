using ILGPU.Algorithms;
using ILGPU.Runtime;
using NSS;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class Input2D : Layer2D
    {
        public override LayerType ActivationType => LayerType.None;
        public override LayerConnectedness Connectedness => LayerConnectedness.Full;
        public Input2D(Size2D size, Size2D previous) : base(size, previous) { }

        public override void Activate(Layer previous, Span<float> input, Span<float> output)
        {
            input.CopyTo(output); 
        }

        public override void Derivate(Span<float> input, Span<float> output)
        {
            input.CopyTo(output); 
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            throw new NotImplementedException();
        }
    }
}
