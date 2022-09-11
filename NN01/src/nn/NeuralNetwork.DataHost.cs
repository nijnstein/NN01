using ILGPU;
using ILGPU.Runtime;
using NSS;
using System.Drawing;

namespace NN01
{
    public partial class NeuralNetwork
    {

        public struct nn3
        {
            public const int hiddenlayercount = 3;
            public int samplecount;
            public int samplesize;
            public int populationcount;
            public int size0;
            public int size1;
            public int size2;
            public int classcount => size2;
            public ArrayView3D<float, Stride3D.DenseXY> n0;
            public ArrayView3D<float, Stride3D.DenseXY> w0;
            public ArrayView2D<float, Stride2D.DenseX> b0;
            public ArrayView2D<float, Stride2D.DenseX> g0;
            public ArrayView2D<float, Stride2D.DenseX> d0;
            public ArrayView3D<float, Stride3D.DenseXY> dw0;
            public ArrayView2D<float, Stride2D.DenseX> db0;

            public ArrayView3D<float, Stride3D.DenseXY> n1;
            public ArrayView3D<float, Stride3D.DenseXY> w1;
            public ArrayView2D<float, Stride2D.DenseX> b1;
            public ArrayView2D<float, Stride2D.DenseX> g1;
            public ArrayView2D<float, Stride2D.DenseX> d1;
            public ArrayView3D<float, Stride3D.DenseXY> dw1;
            public ArrayView2D<float, Stride2D.DenseX> db1;

            public ArrayView3D<float, Stride3D.DenseXY> n2;
            public ArrayView3D<float, Stride3D.DenseXY> w2;
            public ArrayView2D<float, Stride2D.DenseX> b2;
            public ArrayView2D<float, Stride2D.DenseX> g2;
            public ArrayView2D<float, Stride2D.DenseX> d2;
            public ArrayView3D<float, Stride3D.DenseXY> dw2;
            public ArrayView2D<float, Stride2D.DenseX> db2;

            public ArrayView2D<float, Stride2D.DenseX> fittness;
            public bool softmax = false;
            public bool batch = false;

            public nn3(NetworkPopulationDataHost data)
            {
                samplecount = data.SampleCount;
                samplesize = data.SampleSize;
                populationcount = data.PopulationCount;

                size0 = data.size0;
                size1 = data.size1;
                size2 = data.size2;

                n0 = data.gpu_n0.View;
                w0 = data.gpu_w0.View;
                b0 = data.gpu_b0.View;
                g0 = data.gpu_g0.View;
                d0 = data.gpu_d0.View;
                dw0 = data.gpu_dw0.View;
                db0 = data.gpu_db0.View;
                n1 = data.gpu_n1.View;
                w1 = data.gpu_w1.View;
                b1 = data.gpu_b1.View;
                g1 = data.gpu_g1.View;
                d1 = data.gpu_d1.View;
                dw1 = data.gpu_dw1.View;
                db1 = data.gpu_db1.View;
                n2 = data.gpu_n2.View;
                w2 = data.gpu_w2.View;
                b2 = data.gpu_b2.View;
                g2 = data.gpu_g2.View;
                d2 = data.gpu_d2.View;
                dw2 = data.gpu_dw2.View;
                db2 = data.gpu_db2.View;

                fittness = data.gpu_fittness.View;

                softmax = false;
                batch = true;
            }
        }


        /// <summary>
        /// some holder for all buffers needed to train a population of networks of 4 layers
        /// - input layer is not included, n0 is the first hidden layer, n3 the output 
        /// </summary>
        public class NetworkPopulationDataHost  : IDisposable 
        {
            /// <summary>
            /// doesnt include input layer 
            /// </summary>
            const int LayerCount = 3; 
            public int ClassCount;
            public int SampleCount;
            public int SampleSize;
            public int PopulationCount;

            public int size0;
            public int size1;
            public int size2;

            public bool isAllocated = false;

            // > layer0
            // neuron [populationIndex, neuronindex] 
            public float[,,] n0;

            // weight [populationIndex, size, prevsize] 
            public float[,,] w0;

            // bias   [populationIndex, neuron] 
            public float[,] b0;

            // backward activation [populationIndex, neuronIndex]
            public float[,] g0;

            // delta      [populationIndex, neuronIndex]
            public float[,] d0;

            // batch: weightdelta [populationIndex, size, prevsize]   (online would be 4d for index of sample)
            public float[,,] dw0;

            // batch: biasdelta  [populationIndex, neuronIndex]  (online 3d)
            public float[,] db0;

            // > layer1
            public float[,,] n1;
            public float[,,] w1;
            public float[,] b1;
            public float[,] g1;
            public float[,] d1;
            public float[,,] dw1;
            public float[,] db1;

            // > layer2
            public float[,,] n2;
            public float[,,] w2;
            public float[,] b2;
            public float[,] g2;
            public float[,] d2;
            public float[,,] dw2;
            public float[,] db2;

            // fittness   [populationIndex, sampleIndex]
            public float[,] fittness;

            public MemoryBuffer3D<float, Stride3D.DenseXY> gpu_n0;
            public MemoryBuffer3D<float, Stride3D.DenseXY>gpu_w0;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_b0;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_g0;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_d0;
            public MemoryBuffer3D<float, Stride3D.DenseXY> gpu_dw0;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_db0;
            public MemoryBuffer3D<float, Stride3D.DenseXY> gpu_n1;
            public MemoryBuffer3D<float, Stride3D.DenseXY> gpu_w1;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_b1;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_g1;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_d1;
            public MemoryBuffer3D<float, Stride3D.DenseXY> gpu_dw1;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_db1;
            public MemoryBuffer3D<float, Stride3D.DenseXY> gpu_n2;
            public MemoryBuffer3D<float, Stride3D.DenseXY> gpu_w2;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_b2;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_g2;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_d2;
            public MemoryBuffer3D<float, Stride3D.DenseXY> gpu_dw2;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_db2;
            public MemoryBuffer2D<float, Stride2D.DenseX> gpu_fittness;


            public NetworkPopulationDataHost(NeuralNetwork nn, int sampleCount, int populationCount)
            {
                PopulationCount = populationCount;
                SampleCount = sampleCount;
                SampleSize = nn.Input.Size;
                ClassCount = nn.Output.Size;
                size0 = nn.layers[1].Size;
                size1 = nn.layers[2].Size;
                size2 = nn.layers[3].Size; // == classcount

                n0  = new float[PopulationCount, SampleCount, size0];
                w0  = new float[PopulationCount, size0, SampleSize];
                b0  = new float[PopulationCount, size0];
                g0  = new float[PopulationCount, SampleSize]; // size0]; 
                d0  = new float[PopulationCount, size0];
                dw0 = new float[PopulationCount, size0, SampleSize];
                db0 = new float[PopulationCount, size0];
                
                n1   = new float[PopulationCount, SampleCount, size1];
                w1   = new float[PopulationCount, size1, size0];
                b1   = new float[PopulationCount, size1];
                g1   = new float[PopulationCount, size0];// size1]; 
                d1   = new float[PopulationCount, size1];
                dw1  = new float[PopulationCount, size1, size0];
                db1  = new float[PopulationCount, size1];
                                
                n2   = new float[PopulationCount, SampleCount, size2];
                w2   = new float[PopulationCount, size2, size1];
                b2   = new float[PopulationCount, size2];
                g2   = new float[PopulationCount, size1]; //size2
                d2   = new float[PopulationCount, size2];
                dw2  = new float[PopulationCount, size2, size1];
                db2  = new float[PopulationCount, size2];

                fittness = new float[PopulationCount, SampleCount];
            }

            public void AllocateGPU(Accelerator acc)
            {
                gpu_n0  = acc.Allocate3DDenseXY(n0);
                gpu_w0  = acc.Allocate3DDenseXY(w0);
                gpu_b0  = acc.Allocate2DDenseX(b0);
                gpu_g0  = acc.Allocate2DDenseX(g0);
                gpu_d0  = acc.Allocate2DDenseX(d0);
                gpu_dw0 = acc.Allocate3DDenseXY(dw0);
                gpu_db0 = acc.Allocate2DDenseX(db0); 
                gpu_n1  = acc.Allocate3DDenseXY(n1);
                gpu_w1  = acc.Allocate3DDenseXY(w1);
                gpu_b1  = acc.Allocate2DDenseX(b1);
                gpu_g1  = acc.Allocate2DDenseX(g1);
                gpu_d1  = acc.Allocate2DDenseX(d1);
                gpu_dw1 = acc.Allocate3DDenseXY(dw1);
                gpu_db1 = acc.Allocate2DDenseX(db1);
                gpu_n2 = acc.Allocate3DDenseXY(n2);
                gpu_w2  = acc.Allocate3DDenseXY(w2);
                gpu_b2  = acc.Allocate2DDenseX(b2);
                gpu_g2  = acc.Allocate2DDenseX(g2);
                gpu_d2  = acc.Allocate2DDenseX(d2);
                gpu_dw2 = acc.Allocate3DDenseXY(dw2);
                gpu_db2 = acc.Allocate2DDenseX(db2);
                gpu_fittness = acc.Allocate2DDenseX(fittness);
                isAllocated = true;
            }

            public void ReleaseGPU()
            {
                gpu_n0.Dispose();
                gpu_w0.Dispose();
                gpu_b0.Dispose();
                gpu_g0.Dispose();
                gpu_d0.Dispose();
                gpu_dw0.Dispose();
                gpu_db0.Dispose();
                gpu_n1.Dispose();
                gpu_w1.Dispose();
                gpu_b1.Dispose();
                gpu_g1.Dispose();
                gpu_d1.Dispose();
                gpu_dw1.Dispose();
                gpu_db1.Dispose();
                gpu_n2.Dispose();
                gpu_w2.Dispose();
                gpu_b2.Dispose();
                gpu_g2.Dispose();
                gpu_d2.Dispose();
                gpu_dw2.Dispose();
                gpu_db2.Dispose();
                gpu_fittness.Dispose();
                isAllocated = false;
            }


            public void Dispose()
            {
                if(!isAllocated) ReleaseGPU(); 
            }

            public nn3 GetGPUBuffer()
            {
                return new nn3(this); 
            }



            internal void InitializeWeightAndBias(Accelerator accelerator, IRandom random)
            {
              //  InitializeDistribution(LayerInitializationType.HeNormal, w0.AsSpan<float>(), random, SampleSize);
              //  InitializeDistribution(LayerInitializationType.HeNormal, w1.AsSpan<float>(), random, n0.GetLength(1));
              //  InitializeDistribution(LayerInitializationType.HeNormal, w2.AsSpan<float>(), random, n1.GetLength(1));
              //
              //  InitializeDistribution(LayerInitializationType.dot01, b0.AsSpan<float>(), random, SampleSize);
              //  InitializeDistribution(LayerInitializationType.dot01, b1.AsSpan<float>(), random, b0.GetLength(1));
              //  InitializeDistribution(LayerInitializationType.dot01, b2.AsSpan<float>(), random, b1.GetLength(1));
            }



            /// <summary>
            /// synchronize weight, bias and delta buffers from gpu to cpu and copy that into the network list 
            /// </summary>
            internal void SyncFromGPUTo(List<NeuralNetwork> networks)
            {
                gpu_w0.CopyToCPU(w0);
                gpu_b0.CopyToCPU(b0);
                gpu_dw0.CopyToCPU(dw0);
                gpu_db0.CopyToCPU(db0);

                gpu_w1.CopyToCPU(w1);
                gpu_b1.CopyToCPU(b1);
                gpu_dw1.CopyToCPU(dw1);
                gpu_db1.CopyToCPU(db1);

                gpu_w2.CopyToCPU(w2);
                gpu_b2.CopyToCPU(b2);
                gpu_dw2.CopyToCPU(dw2);
                gpu_db2.CopyToCPU(db2);

                
                for (int i = 0; i < networks.Count; i++) 
                {
                    if (networks[i].layers[1].BiasDeltas == null || networks[i].layers[1].WeightDeltas == null)
                        networks[i].layers[1].InitializeDeltaBuffers();
                    if (networks[i].layers[2].BiasDeltas == null || networks[i].layers[2].WeightDeltas == null)
                        networks[i].layers[2].InitializeDeltaBuffers();
                    if (networks[i].layers[3].BiasDeltas == null || networks[i].layers[3].WeightDeltas == null)
                        networks[i].layers[3].InitializeDeltaBuffers();

                    w0.AsSpan3D<float>().Span2D(i).Span.CopyTo(networks[i].layers[1].Weights.AsSpan<float>());
                    w1.AsSpan3D<float>().Span2D(i).Span.CopyTo(networks[i].layers[2].Weights.AsSpan<float>());
                    w2.AsSpan3D<float>().Span2D(i).Span.CopyTo(networks[i].layers[3].Weights.AsSpan<float>());
                    
                    dw0.AsSpan3D<float>().Span2D(i).Span.CopyTo(networks[i].layers[1].WeightDeltas.AsSpan<float>());
                    dw1.AsSpan3D<float>().Span2D(i).Span.CopyTo(networks[i].layers[2].WeightDeltas.AsSpan<float>());
                    dw2.AsSpan3D<float>().Span2D(i).Span.CopyTo(networks[i].layers[3].WeightDeltas.AsSpan<float>());

                    b0.AsSpan2D<float>().Row(i).CopyTo(networks[i].layers[1].Biases);
                    b1.AsSpan2D<float>().Row(i).CopyTo(networks[i].layers[2].Biases);
                    b2.AsSpan2D<float>().Row(i).CopyTo(networks[i].layers[3].Biases);

                    db0.AsSpan2D<float>().Row(i).CopyTo(networks[i].layers[1].BiasDeltas);
                    db1.AsSpan2D<float>().Row(i).CopyTo(networks[i].layers[2].BiasDeltas);
                    db2.AsSpan2D<float>().Row(i).CopyTo(networks[i].layers[3].BiasDeltas);
                }
            }
        }
    }
}

