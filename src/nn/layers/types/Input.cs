using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class InputLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.None;
        public InputLayer(int size) : base(size, 0, LayerInitializer.Zeros, LayerInitializer.Zeros) { }

        public override void Activate(Layer previous)
        {
            throw new NotImplementedException();
        }

        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            throw new NotImplementedException();
        }
    }
}
