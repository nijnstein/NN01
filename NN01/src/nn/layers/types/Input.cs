using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class InputLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.None;
        public InputLayer(int size) : base(size, 0, Distribution.Zeros, Distribution.Zeros) { }

        public override void Activate(Layer previous)
        {
            throw new NotImplementedException();
        }

        public override void Derivate(Span<float> output)
        {
            throw new NotImplementedException(); 
        }

        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            throw new NotImplementedException();
        }
    }
}
