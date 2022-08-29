using ILGPU;
using ILGPU.IR.Values;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public struct LayerInfo 
    {
        public int Size = 0;
        public int LayerIndex = 0;
        public int LayerCount = 0;

        public LayerActivationFunction ActivationType = LayerActivationFunction.None;
        public LayerConnectedness Connectedness = LayerConnectedness.Full;
        public Distribution WeightInitializer = Distribution.Random;
        public Distribution BiasInitializer = Distribution.Random;


        public bool IsInput => LayerIndex == 0 && LayerCount > 0; 
        public bool IsOutput => LayerIndex == LayerCount - 1; 

        public const int BufferCount = 6;
        public const int NeuronBufferId = 0;
        public const int GammaBufferId = 1;
        public const int WeightBufferId = 2;
        public const int BiasBufferId = 3;
        public const int WeightDeltaBufferId = 4;
        public const int BiasDeltaBufferId = 5;

        public LayerInfo()
        {
        }
        public int CalculateLayerBufferSize(LayerInfo next = default)
        {
            if (IsOutput)
            {
                return
                    // neurons + gamma 
                    Size.Align256() * 2;
            }
            else
            {
                int s256 = Size.Align256(); 
                return
                    // neurons + backward activation probs + gamma 
                    (Size.Align256() * 3) +
                    // weights + bias 
                    (Size * next.Size).Align256() * 2 + next.Size.Align256() * 2; 
            }
        }

        /// <summary>
        /// enumerates unaligned sizes of individual buffers for this layer 
        /// </summary>
        /// <param name="next"></param>
        /// <returns></returns>
        public IEnumerable<int> EnumerateBufferSizes(LayerInfo next = default)
        {
            if (IsOutput)
            {
                yield return Size; // neurons 
                yield return Size; // gamma 
            }
            else
            {
                yield return Size; // neurons;
                yield return Size; // gamma;
                yield return Size * next.Size ; // weights;
                yield return next.Size; // bias;
                yield return Size * next.Size; // delta w;
                yield return next.Size; // delta bias;
            }
        }
    }
}
