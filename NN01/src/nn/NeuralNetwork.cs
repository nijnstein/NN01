using ILGPU;
using ILGPU.IR.Values;
using ILGPU.Runtime;
using Microsoft.Win32.SafeHandles;
using NSS;
using NSS.GPU;
using NSS.Neural;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Net.Sockets;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Transactions;
using static ILGPU.Backends.VariableAllocator;

namespace NN01
{
    public partial class NeuralNetwork : IComparable
    {
        private Layer[] layers;

        public float Fitness = 0;
        public float Cost = 0;
        public float CostDelta = 0;
        public int LayerCount => layers == null ? 0 : layers.Length;
        public Layer Input => this[0];
        public Layer Output => this[layers.Length - 1];

        public Layer this[int index]
        {
            get
            {
                if (layers == null) throw new NotInitializedException();
                if (index < 0 || index >= layers.Length) throw new IndexOutOfRangeException($"index {index} is not a valid layer index, layercount = {layers.Length}"); 
                return layers[index];
            }
        }

        public NeuralNetwork(int[] layerSizes, LayerActivationFunction[] activations, bool softmax = false, IRandom random = null)
        {
            if (layerSizes == null) throw new ArgumentNullException("layerSizes");
            if (activations == null) throw new ArgumentNullException("activations");
            if (layerSizes.Length <= 2 || activations.Length <= 1 || layerSizes.Length != activations.Length + 1) throw new ArgumentOutOfRangeException("there must be at least 3 layers and activations must be 1 index shorter as they sit between layers");

            if (random == null)
            {
                random = new CPURandom(RandomDistributionInfo.Uniform(0.5f, 0.5f));
            }

            layers = new Layer[layerSizes.Length];

            layers[0] = CreateLayer(layerSizes[0]);
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i] = CreateLayer(layerSizes[i], layerSizes[i - 1], activations[i - 1], LayerInitializationType.Default, LayerInitializationType.Default, i == layers.Length - 1 ? softmax : false, false, random);
            }
        }
        public NeuralNetwork(Layer[] layers)
        {
            if (layers == null) throw new ArgumentNullException("layers"); 
            this.layers = layers; 
        }

        public NeuralNetwork(Stream stream) : this(new BinaryReader(stream))
        {
        }

        public NeuralNetwork(BinaryReader r)
        {
            if (r == null) throw new ArgumentNullException("binaryreader r");

            string magic = r.ReadString();
            if (string.Compare(magic, "NN01", false) != 0) throw new InvalidFormatException("stream does not contain data in correct format for reading in a neural network (magic marker not present)");

            int layerCount = r.ReadInt32();

            Fitness = r.ReadSingle();
            Cost = r.ReadSingle();

            layers = new Layer[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                layers[i] = r.ReadLayer();
            }
        }

        public NeuralNetwork(NeuralNetwork other, bool clone = true)
        {
            layers = new Layer[other.layers.Length];

            layers[0] = CreateLayer(other.layers[0].Size);

            for (int i = 1; i < other.layers.Length; i++)
            {
                layers[i] = CreateLayer(
                    other.layers[i].Size, 
                    other.layers[i - 1].Size,
                    other.layers[i].ActivationType,
                    other.layers[i].WeightInitializer,
                    other.layers[i].BiasInitializer,
                    other.layers[i].Softmax,
                    clone
                );
            }

            if (clone)
            {
                other.DeepClone(this);
            }
        }

        public NeuralNetwork(NeuralNetwork other, IRandom random)
        {
            layers = new Layer[other.layers.Length];

            layers[0] = CreateLayer(other.layers[0].Size);

            for (int i = 1; i < other.layers.Length; i++)
            {
                layers[i] = CreateLayer(
                    other.layers[i].Size,
                    other.layers[i - 1].Size,
                    other.layers[i].ActivationType,
                    other.layers[i].WeightInitializer,
                    other.layers[i].BiasInitializer,
                    other.layers[i].Softmax,
                    false,
                    random
                );
            }
        }


        internal static Layer CreateLayer(int size, int previousSize = 0, LayerActivationFunction activationType = LayerActivationFunction.None, LayerInitializationType weightInit = LayerInitializationType.Default, LayerInitializationType biasInit = LayerInitializationType.Default, bool softmax = false, bool skipInitializers = false, IRandom random = null)
        {
            if(previousSize == 0)
            {
                return new InputLayer(size); 
            }

            switch (activationType)
            {
                case LayerActivationFunction.ReLU:
                    return new ReLuLayer(size, previousSize, weightInit, biasInit, softmax, skipInitializers, random);

                case LayerActivationFunction.LeakyReLU:
                    return new LeakyReLuLayer(size, previousSize, weightInit, biasInit, softmax, skipInitializers, random);

                case LayerActivationFunction.Tanh:
                    return new TanhLayer(size, previousSize, weightInit, biasInit, softmax, skipInitializers, random);

                case LayerActivationFunction.Sigmoid:
                    return new SigmoidLayer(size, previousSize, weightInit, biasInit, softmax, skipInitializers, random);
                    
                case LayerActivationFunction.Swish:
                    return new SwishLayer(size, previousSize, weightInit, biasInit, softmax, skipInitializers, random);

                case LayerActivationFunction.Binary:
                    return new BinaryLayer(size, previousSize, weightInit, biasInit, softmax, skipInitializers, random);
             }

            throw new InvalidLayerException($"cannot create layer, size: {size}, previous size: {previousSize}, activation: {activationType}, weight: {weightInit}, bias: {biasInit}, skip init: {skipInitializers}"); 
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            bool softmax = (Output != null) && (Output.Softmax);

            sb.Append("[");
            for (int i = 0; i < layers.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append($"-{layers[i].ActivationType}-");
                }
                sb.Append($"{layers[i].Size}");
            }
            if (softmax)
            {
                sb.Append($"-softmax-{Output!.Size}"); 
            }
            sb.Append("]");


            return sb.ToString();
        }


        /// <summary>
        /// feed forward a single sample: inputs -> outputs.
        /// - fills: output.neurons
        /// </summary>
        public Span<float> FeedForward(Span<float> inputs)
        {
            // copy input neurons  
            inputs.CopyTo(Input.Neurons);
            
            // propagate state through layers 
            for (int i = 1; i < layers.Length; i++)
            {
                // activate neurons in current layer from state of previous layer 
                layers[i].Activate(layers[i - 1]);
            }

            // apply softmax normalization on the last layer (actual) if enabled 
            if (layers[layers.Length - 1].Softmax)
            {
                MathEx.Softmax(Output.Neurons, Output.Neurons, true);
            }

            // return the output neuron state 
            return Output.Neurons;
        }

        static public float CalculateError(float ideal, float actual) => .5f * (ideal - actual).Square(); 
        static public float CalculateError(Span<float> ideal, Span<float> actual) => 0.5f * Intrinsics.SumSquaredDifferences(ideal, actual); 
        static public float CalculateActivationDelta(float ideal, float actual) => -(ideal - actual);

 
        /// <summary>
        /// 
        ///  Feed forward all samples in the set  
        ///  
        ///  - uses shuffled indices 
        ///  - updates sample error 
        ///  - stored actual|output activation in samples.Actual buffer instead of network output neurons
        ///  
        ///  NOTE: 
        ///  the gpu feeds all populations in one action while the cpu method feeds only 1
        /// 
        /// </summary>
        /// <returns>total input set error</returns>
        public float FeedForward(SampleSet samples, int populationIndex)
        {
            float error = 0f;

            // reset delta & gamma buffers 
            
            for (int i = 0; i < layers.Length - 1; i++)
            {
                layers[i].Delta.Zero();
                layers[i].Gamma.Zero(); 
            }

                // Fallback: CPU-AVX
                // - handles 1 POPULATION at a time 
                for (int sampleIndex = 0; sampleIndex < samples.SampleCount; sampleIndex++)
                {
                    Span<float> sample = samples.ShuffledData(sampleIndex);
                    Span<float> actual = samples.ShuffledActual(populationIndex, sampleIndex);
                    Span<float> ideal = samples.ShuffledExpectation(sampleIndex);

                    // activate first hidden layer from sample data
                    layers[1].Activate(layers[0], sample, layers[1].Neurons);

                    // propagate state through hidden layers 
                    for (int i = 2; i < layers.Length - 1; i++)
                    {
                        layers[i].Activate(layers[i - 1]);
                    }

                    // then activate the last into the actual buffer
                    layers[layers.Length - 1].Activate(layers[layers.Length - 2], layers[layers.Length - 2].Neurons, actual);

                    // apply softmax normalization on the last layer (actual) if enabled 
                    if (layers[layers.Length - 1].Softmax)
                    {
                        MathEx.Softmax(actual, actual, true);
                    }

                    // calc error on sample
                    float sampleError = CalculateError(ideal, actual);
                
                  //  actual.CopyTo(layers[layers.Length - 1].Neurons);

                    error += sampleError;
                    samples.SetShuffledError(populationIndex, sampleIndex, sampleError);
                }
            
            return error; 
        }

     

        public void BackPropagateBatchCPU
        (
            int populationIndex,
            SampleSet trainingSet,
            float weightLearningRate = 0.01f, float biasLearningRate = 0.01f, float momentum = 1f, float weightCost = 0.00001f, float minCostDelta = 0.0000001f,
            IRandom random = null
         )
        {
            // feed forward all samples and get the total error 
            float totalError = (FeedForward(trainingSet, populationIndex) / (trainingSet.SampleSize * trainingSet.SampleCount * trainingSet.Variance));

            // calculate total error change foreach output
            Span3D<float> actual = trainingSet.Actual.AsSpan3D<float>();
            Span2D<float> expected = trainingSet.Expectation.AsSpan2D<float>();

            for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                if (layers[layerIndex].WeightDeltas == null)
                {
                    layers[layerIndex].InitializeDeltaBuffers();
                }
            }

            // calculate error on each weight going from  prev.neuron[E].weight[i] -> out[i]
            // known as the delta rule: -(target - out) * derivative(out) * out 
            for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                layers[layerIndex].Delta.Zero();
                

                for (int sampleIndex = 0; sampleIndex < trainingSet.SampleCount; sampleIndex++)
                {
                    int shuffledIndex = trainingSet.ShuffledSample(sampleIndex).Index;

                    Span<float> a = layerIndex == layers.Length - 1 ? actual.Row(populationIndex, shuffledIndex) : layers[layerIndex].Neurons.AsSpan();
                    Span<float> e = layerIndex == layers.Length - 1 ? expected.Row(shuffledIndex) : layers[layerIndex].Gamma; //gamma[layerIndex];

                    if (layers[layerIndex].Softmax)
                    {
                        // if softmax is enabled take its derivate first 
                        //MathEx.SoftmaxDerivative(a, a);
                    }

                    // calc derivate if actual output == gamma 
                    layers[layerIndex].Derivate(a, layers[layerIndex].Gamma);

                    // calc error delta with respect to derivate of actual output 
                    float inverse = 1 / (trainingSet.SampleCount * a.Length);
                    for (int i = 0; i < a.Length; i++)
                    {
                        layers[layerIndex].Delta[i] +=
                        (
                            inverse
                            *
                            CalculateActivationDelta(e[i], a[i]) //   -(ideal - actual)
                            *
                            layers[layerIndex].Gamma[i]
                        );
                    }

                    // calc the new actual for this sample for the previous layer
             /*      layers[layerIndex - 1].Gamma.Zero();
                    for (int j = 0; j < layers[layerIndex - 1].Size; j++)
                    {
                        for (int i = 0; i < layers[layerIndex].Size; i++)
                        {
                            layers[layerIndex - 1].Gamma[j] += layers[layerIndex].Delta[i] * layers[layerIndex].Weights[i, j];
                        }
                    }*/ 
                }
                Intrinsics.MultiplyScalar(layers[layerIndex].Delta, 1f / layers[layerIndex].Size);
            }

            /* for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
             {
                 // dit zou in de lus hierboven moeten om gamma voor volgende iteratie klaar te maken.... 
                 ///   calc hidden activation error ..........we trainen 1 laag niet goed volgens mij zo
                 layers[layerIndex - 1].Gamma.Zero();

                 // calc hidden activation error of previous layer using gamma 
                 for (int j = 0; j < layers[layerIndex - 1].Size; j++)
                 {
                     for (int i = 0; i < layers[layerIndex].Size; i++)
                     {
                         layers[layerIndex - 1].Gamma[j] += layers[layerIndex].Delta[i] * layers[layerIndex].Weights[i, j];
                     }
                 }
             }*/


            Parallel.For(layers.Length - 1, 1, (layerIndex) =>
            //for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                // update biases 
                for (int j = 0; j < layers[layerIndex].Size; j++)
                {
                    float delta = layers[layerIndex].Gamma[j] * biasLearningRate;
                    layers[layerIndex].Biases[j] -= delta + (layers[layerIndex].BiasDeltas[j] * momentum);
                    layers[layerIndex].BiasDeltas[j] = delta;
                }

                // update weights from error 
                for (int i = 0; i < layers[layerIndex].Size; i++)
                {
                    for (int j = 0; j < layers[layerIndex - 1].Size; j++)
                    {
                        float wDelta = weightLearningRate * layers[layerIndex].Delta[i] * layers[layerIndex - 1].Gamma[j];

                        layers[layerIndex].Weights[i, j] -=
                            // current weight to error delta
                            wDelta
                            // momentum 
                            + (layers[layerIndex].WeightDeltas![i, j] * momentum)
                            // strengthen learned 
                            + (weightLearningRate * (layers[layerIndex - 1].Gamma[j] - weightCost * layers[layerIndex].Weights[i, j]));

                        layers[layerIndex].WeightDeltas[i, j] = wDelta;
                    }
                }
            });

            Cost = Math.Abs(totalError);
        }

            /// <summary>
            /// backtrack state through derivates of the activation minimizing cost/error
            /// </summary>                
        public void BackPropagateOnline(
            Span<float> inputs, Span<float> expected, float[][] gamma, 
            float weightLearningRate = 0.01f, float biasLearningRate = 0.01f, float momentum = 1f, float weightCost = 0.00001f, float minCostDelta = 0.0000001f)
        {
            // initialize neurons from input patterns
            FeedForward(inputs);

            float cost = 0;
            float[] delta = new float[Output.Neurons.Length];
            for (int i = 0; i < delta.Length; i++)
            {
                // precalculate delta 
                delta[i] = Output.Neurons[i] - expected[i];
                                        
                // calculate cost of network 
                cost += delta[i] * delta[i];
            }
            CostDelta = MathF.Abs(Cost - cost);
            if (Cost > 0 && CostDelta < minCostDelta) return; 
            Cost = cost;

            // calculate 'gamma' for the last layer
            // - gamma == derivate activation of previous neuron layer

            if (layers[layers.Length - 1].Softmax)
            {
                // if softmax is enabled take its derivate first 
                MathEx.SoftmaxDerivative(Output.Neurons, Output.Neurons);
            }

            Output.CalculateGamma(delta, gamma[layers.Length - 1], Output.Neurons);
  
            // calculate new weights and biases for the last layer in the network 
            Output.Update(layers[layers.Length - 2], gamma[layers.Length - 1], weightLearningRate, biasLearningRate, momentum, weightCost);

            // now propagate backward 
            for (int i = layers.Length - 2; i > 0; i--)
            {
                // update gamma from layer weights and current gamma on output
                for (int j = 0; j < layers[i].Size; j++)
                {
                    // 
                    //  get weighed sum of gamma * previous neurons 
                    // 
                    int k = 0;
                    gamma[i][j] = 0;
                    if (Avx.IsSupported && gamma[i + 1].Length > 15)
                    {
                        Span<Vector256<float>> g = MemoryMarshal.Cast<float, Vector256<float>>(gamma[i + 1]);
                        Vector256<float> g0 = Vector256<float>.Zero; 
                        int l = 0; 
                        while(k < (gamma[i + 1].Length & ~7))
                        {
                            g0 = Intrinsics.MultiplyAdd(g[l] , Vector256.Create
                                (
                                    // could use gather.. 
                                    layers[i + 1].Weights[k + 0, j],
                                    layers[i + 1].Weights[k + 1, j],
                                    layers[i + 1].Weights[k + 2, j],
                                    layers[i + 1].Weights[k + 3, j],
                                    layers[i + 1].Weights[k + 4, j],
                                    layers[i + 1].Weights[k + 5, j],
                                    layers[i + 1].Weights[k + 6, j],
                                    layers[i + 1].Weights[k + 7, j]
                                ), g0);
                            k += 8;
                            l++;
                        }
                        gamma[i][j] = Intrinsics.HorizontalSum(g0); 
                    }
                    while(k < gamma[i + 1].Length)
                    {
                        gamma[i][j] += gamma[i + 1][k] * layers[i + 1].Weights[k, j];
                        k++; 
                    }

                    if (layers[i].Softmax)
                    {
                        // if softmax is enabled take its derivate first 
                        MathEx.SoftmaxDerivative(gamma[i], gamma[i]);
                    }

                    //
                    // Calculate the new gamma from the activation derivate
                    //
                    layers[i].CalculateGamma(gamma[i], gamma[i], layers[i].Neurons);
                }

                // update layer weights and biases
                layers[i].Update(layers[i - 1], gamma[i], weightLearningRate, biasLearningRate, momentum, weightCost); 
            }
        }

        /// <summary>
        /// randomly mutate some of the worst networks
        /// </summary>
        public int Mutate(IRandom random, float chance, float weightRange, float biasRange) 
        {
            int mutationCount = 0;

            Parallel.For(1, LayerCount, (i) =>
            //for (int i = 1; i < LayerCount; i++)
            {
                Span<float> rnd = random.Span(layers[i].Biases.Length);

                for (int j = 0; j < layers[i].Biases.Length; j++)
                {
                    bool b = rnd[j] > chance; // random.NextSingle() > chance; 

                    layers[i].Biases[j] =
                        b
                        ?
                        layers[i].Biases[j] += (biasRange + biasRange) * rnd[rnd.Length - j - 1] - biasRange    //rnd(-biasRange, biasRange)
                        :
                        layers[i].Biases[j];

                    mutationCount += b ? 1 : 0;  // (int)b  
                }
                //}
                //for (int i = 1; i < LayerCount; i++)
                //{
                for (int j = 0; j < layers[i].Weights.GetLength(0); j++)
                {
                    /*Span<float>*/ rnd = random.Span(layers[i].Weights.GetLength(1));
                    for (int k = 0; k < layers[i].Weights.GetLength(1); k++)
                    {
                        bool b = rnd[k] > chance;

                        layers[i].Weights[j, k] =
                            b
                            ?
                            layers[i].Weights[j, k] += (weightRange + weightRange) * rnd[rnd.Length - k - 1] - weightRange
                            :
                            layers[i].Weights[j, k];

                        mutationCount += b ? 1 : 0;
                    }
                }
            }); 

            return mutationCount;
        }

        /// <summary>
        /// reset weights and biases
        /// </summary>
        public void Reset()
        { 
        }

        public void DeepClone(NeuralNetwork into)
        {
            into.Cost = Cost;
            into.Fitness = Fitness;

            for (int i = 0; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].Neurons.Length; j++)
                {
                    into.layers[i].Neurons[j] = layers[i].Neurons[j];
                }

                if (!layers[i].IsInput)
                {
                    for (int j = 0; j < layers[i].Biases.Length; j++)
                    {
                        into.layers[i].Biases[j] = layers[i].Biases[j];
                    }
                    for (int j = 0; j < layers[i].Weights.GetLength(0); j++)
                    {
                        for (int k = 0; k < layers[i].Weights.GetLength(1); k++)
                        {
                            into.layers[i].Weights[j, k] = layers[i].Weights[j, k];
                        }
                    }
                }
            }
        }

        public void WriteTo(Stream stream)
        {
            if (stream == null) throw new ArgumentNullException("stream");
            if (!stream.CanWrite) throw new IOException("cannot write to stream, it is non writable"); 

            using(BinaryWriter w = new BinaryWriter(stream))
            {
                w.Write(this); 
            }
        }

        public static NeuralNetwork ReadFrom(Stream stream)
        {
            if (stream == null) throw new ArgumentNullException("stream");
            if (!stream.CanRead) throw new IOException("cannot read from stream, it is not readable");

            using(BinaryReader r = new BinaryReader(stream))
            {
                return r.ReadNeuralNetwork(); 
            }
        }

        public NeuralNetwork DeepCopy()
        {
            return new NeuralNetwork(this);
        }

        public int CompareTo(object? obj)
        {
            return CompareTo((obj as NeuralNetwork)!);
        }

        public int CompareTo(NeuralNetwork other)
        {
            if (other == null) return 1;

            if (Fitness > other.Fitness)
                return 1;
            else if (Fitness < other.Fitness)
                return -1;
            else
                return 0;
        }

        public int CalculateTotalParameterCount()
        {
            int c = 0;

            // input provides no parameters 

            for(int i = 1; i < layers.Length; i++)
            {
                Layer layer = layers[i];
                c +=
                    layer.Size * 2 // neurons + bias
                    +
                    layer.Size * layer.PreviousSize * 1; // weights
            }

            return c; 
        }
    }
}

