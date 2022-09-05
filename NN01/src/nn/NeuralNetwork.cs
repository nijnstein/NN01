using ILGPU.Runtime;
using NSS;
using NSS.GPU;
using NSS.Neural;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.AccessControl;
using System.Text;

namespace NN01
{
    public class NeuralNetwork : IComparable
    {
        private Layer[] layers;

        public float Fitness = 0;
        public float Cost = 0;
        public float CostDelta = 0;

        public Layer this[int index]
        {
            get
            {
                if (layers == null) throw new NotInitializedException();
                if (index < 0 || index >= layers.Length) throw new IndexOutOfRangeException($"index {index} is not a valid layer index, layercount = {layers.Length}"); 
                return layers[index];
            }
        }

        public int LayerCount => layers == null ? 0 : layers.Length;

        public Layer Input => this[0];
        public Layer Output => this[layers.Length - 1];

        public float[][] Gamma { get; set; } = null;

        public NeuralNetwork(int[] layerSizes, LayerActivationFunction[] activations)
        {
            if (layerSizes == null) throw new ArgumentNullException("layerSizes");
            if (activations == null) throw new ArgumentNullException("activations");
            if (layerSizes.Length <= 2 || activations.Length <= 1 || layerSizes.Length != activations.Length + 1) throw new ArgumentOutOfRangeException("there must be at least 3 layers and activations must be 1 index shorter as they sit between layers");

            layers = new Layer[layerSizes.Length];

            layers[0] = CreateLayer(layerSizes[0]);
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i] = CreateLayer(layerSizes[i], layerSizes[i - 1] , activations[i - 1]);
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
                    clone
                );
            }

            if (clone)
            {
                other.DeepClone(this);
            }
        }

        public NeuralNetwork(NeuralNetwork other, GPURandom random)
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
                    false,
                    random
                );
            }
        }


        internal static Layer CreateLayer(int size, int previousSize = 0, LayerActivationFunction activationType = LayerActivationFunction.None, LayerInitializationType weightInit = LayerInitializationType.Default, LayerInitializationType biasInit = LayerInitializationType.Default, bool skipInitializers = false, GPURandom random = null)
        {
            if(previousSize == 0)
            {
                return new InputLayer(size); 
            }

            switch (activationType)
            {
                case LayerActivationFunction.ReLU:
                    return new ReLuLayer(size, previousSize, weightInit, biasInit, skipInitializers, random);

                case LayerActivationFunction.LeakyReLU:
                    return new LeakyReLuLayer(size, previousSize, weightInit, biasInit, skipInitializers, random);

                case LayerActivationFunction.Tanh:
                    return new TanhLayer(size, previousSize, weightInit, biasInit, skipInitializers, random);

                case LayerActivationFunction.Sigmoid:
                    return new SigmoidLayer(size, previousSize, weightInit, biasInit, skipInitializers, random);
                    
                case LayerActivationFunction.Swish:
                    return new SwishLayer(size, previousSize, weightInit, biasInit, skipInitializers, random);

                case LayerActivationFunction.Binary:
                    return new BinaryLayer(size, previousSize, weightInit, biasInit, skipInitializers, random);
            }

            throw new InvalidLayerException($"cannot create layer, size: {size}, previous size: {previousSize}, activation: {activationType}, weight: {weightInit}, bias: {biasInit}, skip init: {skipInitializers}"); 
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("[");
            for (int i = 0; i < layers.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append($"-{layers[i].ActivationType}-");
                }
                sb.Append($"{layers[i].Size}");
            }
            sb.Append("]");
            return sb.ToString();
        }


        /// <summary>
        /// feed forward, inputs -> outputs.
        /// </summary>
        public Span<float> FeedForward(Span<float> inputs)
        {
            // copy input neurons  
            for (int i = 0; i < inputs.Length; i++)
            {
                Input.Neurons[i] = inputs[i];
            }

            
            // propagate state through layers 
            for (int i = 1; i < layers.Length; i++)
            {
                // activate neurons in current layer from state of previous layer 
                layers[i].Activate(layers[i - 1]);
            }

            // return the output neuron state 
            return Output.Neurons;
        }

        public float CalculateError(float ideal, float actual) => .5f * (ideal - actual).Square(); 

        public float CalculateActivationErrorOnOutput(float ideal, float actual) => -(ideal - actual);

        /// <summary>
        /// 
        ///  Feed forward all samples in the set (or minibatch if enabled) 
        ///  - uses shuffled indices 
        ///  - updates sample error 
        ///  - stored actual|output activation in samples.Actual buffer instead of network output neurons
        /// 
        /// </summary>
        /// <returns>total input set error</returns>

        public float FeedForward(SampleSet samples)
        {
            float error = 0f;

            // reset delta & gamma buffers 
            for (int i = 0; i < layers.Length - 1; i++)
            {
                layers[i].Delta.Zero();
                layers[i].Gamma.Zero(); 
            }

            // feed all samples 
            for (int sampleIndex = 0; sampleIndex < samples.SampleCount; sampleIndex++)
            {
                Span<float> sample = samples.ShuffledData(sampleIndex); 
                Span<float> actual = samples.ShuffledActual(sampleIndex);
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

                // calc error on sample 
                float sampleError = 0; 
                for (int i = 0; i < actual.Length; i++)
                {
                    sampleError += CalculateError(ideal[i], actual[i]);
                }
                
                error += sampleError;
                Console.WriteLine(sampleError); 
                samples.SetShuffledError(sampleIndex, sampleError);
            }          
            
            return error; 
        }

  

        public float[] Backward(float[] output)
        {
            Debug.Assert(output.Length == Output.Neurons.Length);

            // copy output into neurons in output layer 
            for (int i = 0; i < output.Length; i++)
            {
                Output.Neurons[i] = output[i];
            }

            // calculate 'gamma' for the last layer
            // - gamma == derivate activation of previous neuron layer
            //Output.CalculateGamma(delta, gamma[layers.Length - 1], Output.Neurons);

            // calculate new weights and biases for the last layer in the network 
            //Output.Update(layers[layers.Length - 2], gamma[layers.Length - 1], weightLearningRate, biasLearningRate, momentum, weightCost);

            // now propagate backward 
            for (int i = layers.Length - 2; i >= 0; i--)
            {
                // update gamma from layer weights and current gamma on output
                for (int j = 0; j < layers[i].Size; j++)
                {
                    Gamma[i][j] = 0;
                    for (int k = 0; k < Gamma[i + 1].Length; k++)
                    {
                        Gamma[i][j] += Gamma[i + 1][k] * layers[i + 1].Weights[k][j];
                    }

                    layers[i].CalculateGamma(Gamma[i], Gamma[i], layers[i].Neurons);
                }
            }

            return Input.Neurons;
        } 

        public float CalcMSE(float[][] s1, float[][] sn)
        {
            float mse = 0f;
            unchecked
            {
                for (int i = 0; i < s1.Length; i++)
                {
                    mse += Intrinsics.SumSquaredDifferences(sn[i], s1[i]);
                }
            }
            return mse;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainingSet"></param>
        /// <param name="gamma"></param>
        /// <param name="weightLearningRate"></param>
        /// <param name="biasLearningRate"></param>
        /// <param name="momentum"></param>
        /// <param name="weightCost"></param>
        /// <param name="minCostDelta"></param>
        public void BackPropagate(
            SampleSet trainingSet,
            float[][] gamma,
            float weightLearningRate = 0.01f, float biasLearningRate = 0.01f, float momentum = 1f, float weightCost = 0.00001f, float minCostDelta = 0.0000001f)
        {
            // shuffle training set 
            trainingSet.ShuffleIndices(Random.Shared);

            // feed forward all samples and get the total error 
            float totalError = FeedForward(trainingSet);

            // calculate total error change foreach output
            Span2D<float> actual = trainingSet.Actual.AsSpan2D<float>();
            Span2D<float> expected = trainingSet.Expectation.AsSpan2D<float>();

            // calculate error on each weight going from  prev.neuron[E].weight[i] -> out[i]
            // known as the delta rule: -(target - out) * derivative(out) * out 

            for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                if (layers[layerIndex].WeightDeltas == null)
                {
                    layers[layerIndex].InitializeDeltaBuffers();
                }
            }

            for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                for (int sampleIndex = 0; sampleIndex < trainingSet.SampleCount; sampleIndex++)
                {
                    int shuffledIndex = trainingSet.ShuffledSample(sampleIndex).Index;

                    Span<float> a = layerIndex == layers.Length - 1 ? actual.Row(shuffledIndex) : layers[layerIndex].Neurons.AsSpan();
                    Span<float> e = layerIndex == layers.Length - 1 ? expected.Row(shuffledIndex) : gamma[layerIndex];

                    // calc derivate if actual output == gamma 
                    layers[layerIndex].Derivate(a, layers[layerIndex].Gamma);

                    // calc error delta with respect to derivate of actual output 
                    for (int i = 0; i < a.Length; i++)
                    {
                        layers[layerIndex].Delta[i] +=
                            (CalculateActivationErrorOnOutput(e[i], layers[layerIndex].Neurons[i])
                            *
                            layers[layerIndex].Gamma[i]);  // e[i]);  //  / trainingSet.SampleCount;
                    }
                }
            }

            for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                //for (int sampleIndex = 0; sampleIndex < trainingSet.SampleCount; sampleIndex++)
                {
                    // calc hidden activation error 
                    for (int j = 0; j < layers[layerIndex - 1].Size; j++)
                    {
                        for (int i = 0; i < layers[layerIndex].Size; i++)
                        {
                            layers[layerIndex - 1].Gamma[j] += layers[layerIndex].Delta[i] * layers[layerIndex].Weights[i][j];
                        }
                    }

                    // update weights from error 
                    for (int i = 0; i < layers[layerIndex].Size; i++)
                    {
                        for (int j = 0; j < layers[layerIndex - 1].Size; j++)
                        {
                            layers[layerIndex].Weights[i][j] -= 0.01f * layers[layerIndex].Delta[i] * layers[layerIndex - 1].Gamma[j];
                        }
                    }

                }
            }

            
            Cost = totalError; 
        }

            /// <summary>
            /// backtrack state through derivates of the activation minimizing cost/error
            /// </summary>                
            public void BackPropagate(
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
                                    layers[i + 1].Weights[k + 0][j],
                                    layers[i + 1].Weights[k + 1][j],
                                    layers[i + 1].Weights[k + 2][j],
                                    layers[i + 1].Weights[k + 3][j],
                                    layers[i + 1].Weights[k + 4][j],
                                    layers[i + 1].Weights[k + 5][j],
                                    layers[i + 1].Weights[k + 6][j],
                                    layers[i + 1].Weights[k + 7][j]
                                ), g0);
                            k += 8;
                            l++;
                        }
                        gamma[i][j] = Intrinsics.HorizontalSum(g0); 
                    }
                    while(k < gamma[i + 1].Length)
                    {
                        gamma[i][j] += gamma[i + 1][k] * layers[i + 1].Weights[k][j];
                        k++; 
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
        public void Mutate(float chance, float weightRange, float biasRange) 
        {
            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < layers[i].Biases.Length; j++)
                {
                    layers[i].Biases[j] = 
                        Random.Shared.NextSingle() > chance
                        ?
                        layers[i].Biases[j] += Random.Shared.Range(-biasRange, biasRange)
                        :
                        layers[i].Biases[j];
                }
            }

            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < layers[i].Weights.Length; j++)
                {
                    for (int k = 0; k < layers[i].Weights[j].Length; k++)
                    {
                        layers[i].Weights[j][k] = 
                            Random.Shared.NextSingle() > chance 
                            ? 
                            layers[i].Weights[j][k] += Random.Shared.Range(-weightRange, weightRange) 
                            :
                            layers[i].Weights[j][k];
                    }
                }
            }
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
                    for (int j = 0; j < layers[i].Weights.Length; j++)
                    {
                        for (int k = 0; k < layers[i].Weights[j].Length; k++)
                        {
                            into.layers[i].Weights[j][k] = layers[i].Weights[j][k];
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
    }
}

