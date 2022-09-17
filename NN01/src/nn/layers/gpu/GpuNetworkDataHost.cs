using ILGPU.Runtime;

namespace NN01
{
    public class GpuNetworkDataHost : IDisposable
    {
        public int PopulationCount;
        public GpuLayerDataHost[] Layers;
        public bool IsAllocated; 

        private GpuNetworkDataHost()
        {
            IsAllocated = false; 
        }
     
        static public GpuNetworkDataHost AllocatePopulation(Accelerator acc, NeuralNetwork network, int populationCount)
        {
            GpuNetworkDataHost host = new GpuNetworkDataHost();
            host.PopulationCount = populationCount; 
            host.Layers = new GpuLayerDataHost[network.LayerCount]; 

            for(int i = 0; i < network.LayerCount; i++)
            {
                GpuLayerDataHost gpuLayer = GpuLayerDataHost.AllocatePopulation(
                    acc, 
                    network[i],
                    populationCount); 

                host.Layers[i] = gpuLayer;  
            }

            host.IsAllocated = true; 
            return host; 
        }

        public gpu_layer_data[] GetView()
        {
            gpu_layer_data[] view = new gpu_layer_data[Layers.Length];
            for(int i = 0; i < Layers.Length; i++)
            {
                bool b = Layers[i].Type.HasParameters();
                view[i] = new gpu_layer_data()
                {
                    Population = PopulationCount,
                    Size = Layers[i].Size,
                    PreviousSize = Layers[i].Size,
                    Type = Layers[i].Type,

                    Neurons = Layers[i].Neurons.View,
                    Delta = Layers[i].Delta.View,
                    Gamma = Layers[i].Gamma.View,

                    Weights = b ? Layers[i].Weights : default,
                    WeightDeltas = b ? Layers[i].WeightDeltas : default,

                    Biases = b ? Layers[i].Biases : default,
                    BiasDeltas = b ? Layers[i].BiasDeltas : default
                };
            }
            return view;
        }

        public void CopyFromCPU(NeuralNetwork[] networks)
        {

        }

        public void CopyFromGPU(NeuralNetwork[] networks)
        {

        }

        public void ReleaseAllocation()
        {
            IsAllocated = false; 
        }

        public void Dispose()
        {
            if (IsAllocated)
            {
                ReleaseAllocation();
            }
        }
    }
}
