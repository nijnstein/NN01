using ILGPU;
using ILGPU.Runtime;

namespace NN01
{
    public partial class NeuralNetwork
    {
        public class NetworkPopulationDataHost
        {
            public int LayerCount;
            public int ClassCount;
            public int SampleCount;
            public int SampleSize;
            
            public class LayerDataHost
            {
                // > layer0
                // neuron [populationIndex, neuronindex] 
                public float[,] n;

                // weight [populationIndex, nextsize, size] 
                public float[,,] w;

                // bias   [populationIndex, neuron] 
                public float[,] b;

                // activation [populationIndex, neuronIndex]
                public float[,] gamma;

                // delta      [populationIndex, neuronIndex]
                public float[,] delta;

                // batch: weightdelta [populationIndex, nextsize, size]   (online would be 4d for index of sample)
                public float[,,] dw;

                // batch: biasdelta  [populationIndex, neuronIndex]  (online 3d)
                public float[,] db;
            }

            // n layers 
            public List<LayerDataHost> Layers = new List<LayerDataHost>();

            // actual     [populationIndex, sampleIndex, classCount]
            public float[,,] actual;

            // cost       [populationIndex, sampleIndex]
            public float[,] error;

            // fittness   [populationIndex, sampleIndex]
            public float[,] fittness;
        
            public NetworkPopulationDataHost(NeuralNetwork nn)
            {

            }

            public struct layerbuffer
            {

            }
            public struct buffer
            {
                MemoryBuffer1D<layerbuffer, Stride1D.Dense> layers; 
            }

            public void GetGPUBuffer(ref buffer buf)
            {

            }

        }
    }
}

