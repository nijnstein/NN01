using ILGPU;
using ILGPU.IR.Values;
using ILGPU.Runtime;
using Microsoft.Win32.SafeHandles;
using NSS;
using NSS.GPU;
using NSS.Neural;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Diagnostics;
using System.Net.Sockets;
using System.Numerics;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Transactions;
using static ILGPU.Backends.VariableAllocator;
using static ILGPU.IR.Analyses.Uniforms;

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

        public NeuralNetwork(int[] layerSizes, LayerType[] activations, IRandom random = null)
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
                layers[i] = CreateLayer(layerSizes[i], layerSizes[i - 1], activations[i - 1], false, random);
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

            layers[0] = CreateLayer(other.layers[0].Size); // always input.. 

            for (int i = 1; i < other.layers.Length; i++)
            {
                layers[i] = CopyLayer(other.layers[i], other.layers[i - 1], clone); 
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
                layers[i] = CopyLayer(other.layers[i], other.layers[i - 1], false, random); 
            }
        }

        public Layer CopyLayer(Layer other, Layer previous, bool skipInit, IRandom random = null)
        {
            if (other.ActivationType == LayerType.Dropout)
            {
                Dropout dropout = new Dropout(other.Size, (other as Dropout).DropoutFactor);
                return dropout;  
            }
            else
            {
                Layer layer = CreateLayer(
                    other.Size,
                    previous != null ? previous.Size : 0,
                    other.ActivationType,
                    skipInit,
                    random
                );
                return layer; 
            }
        }

        public void InitializeLayers(IRandom? random)
        {
            bool ownRandom = random == null;
            if(ownRandom)
            {
                random = new CPURandom(RandomDistributionInfo.Uniform(0, 1f)); 
            }
            for (int i = 1; i < layers.Length; i++)
            {
                if (layers[i] is ParameterLayer layer)
                {
                    layer.InitializeParameters(random);
                }
            }
            if (ownRandom)
            {
                random!.Dispose(); 
            }
        }

        internal static Layer CreateLayer(int size, int previousSize = 0, LayerType activationType = LayerType.None, bool skipInitializers = false, IRandom random = null)
        {
            if(previousSize == 0)
            {
                return new InputLayer(size); 
            }

            switch (activationType)
            {
                case LayerType.Softmax:
                    return new SoftmaxLayer(size); 

                case LayerType.ReLU:
                    return new ReLuLayer(size, previousSize, skipInitializers, random);

                case LayerType.LeakyReLU:
                    return new LeakyReLuLayer(size, previousSize, skipInitializers, random);

                case LayerType.Tanh:
                    return new TanhLayer(size, previousSize, skipInitializers, random);

                case LayerType.Sigmoid:
                    return new SigmoidLayer(size, previousSize, skipInitializers, random);
                    
                case LayerType.Swish:
                    return new SwishLayer(size, previousSize, skipInitializers, random);
             }

            throw new InvalidLayerException($"cannot create layer, size: {size}, previous size: {previousSize}, activation: {activationType}, skip init: {skipInitializers}"); 
        }

        public override string ToString()
        {
            StringBuilder sb = StringBuilderCache.Acquire();

            bool softmax = (Output != null) && (Output.ActivationType == LayerType.Softmax);
            sb.Append('[');
            for (int i = 0; i < layers.Length; i++)
            {
                if (i > 0)
                {
                    switch(layers[i].ActivationType)
                    {
                        case LayerType.Dropout:
                            sb.Append($"-{layers[i].ActivationType}[{(layers[i] as Dropout).DropoutFactor.ToString("0.00")}]-");
                            break; 

                        default:
                            sb.Append($"-{layers[i].ActivationType}-");
                            break; 
                    }
                }
                sb.Append($"{layers[i].Size}");
            }
            sb.Append(']');
            return StringBuilderCache.GetStringAndRelease(ref sb); 
        }


        /// <summary>
        /// feed forward a single sample: inputs -> outputs.
        /// - fills: output.neurons
        /// </summary>
        public Span<float> FeedForward(Span<float> inputs, bool isTest = false)
        {
            // copy input neurons  
            inputs.CopyTo(Input.Neurons);
            
            // propagate state through layers 
            for (int i = 1; i < layers.Length; i++)
            {
                if (isTest && layers[i].ActivationType == LayerType.Dropout)
                {
                    // skip dropouts in tests
                    layers[i].PassThrough(layers[i - 1]); 
                }
                else
                {
                    // activate neurons in current layer from state of previous layer 
                    layers[i].Activate(layers[i - 1]);
                }
            }

            // return the output neuron state 
            return Output.Neurons;
        }

        static public float CalculateError(float ideal, float actual) => .5f * (ideal - actual).Square(); 
        static public float CalculateError(Span<float> ideal, Span<float> actual) => 0.5f * Intrinsics.SumSquaredDifferences(ideal, actual); 
        static public float CalculateActivationDelta(float ideal, float actual) => -(ideal - actual);

        static public Span<float> CalculateDelta(Span<float> output, Span<float> expected, Span<float> delta)
            => Intrinsics.Substract(output, expected, delta);

        static public float CalculateCost(Span<float> delta) => Intrinsics.SumSquares(delta);

        static public float CalculateCrossEntropyDeltaAndCost(Span<float> actual, Span<float> expected, Span<float> delta)
        {
            float sum = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                if (expected[i] == 1)
                {
                    delta[i] = -MathF.Log(actual[i]);
                    sum += delta[i]; 
                }
                else
                {
                    delta[i] = actual[i] - expected[i];
                }
            }
            return sum; 
        }

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

            // - reset delta & gamma buffers 
             
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

                    // calc error on sample
                    float sampleError = CalculateError(ideal, actual);
                
                    // actual.CopyTo(layers[layers.Length - 1].Neurons);

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

            // calculate error on each weight going from  prev.neuron[E].weight[i] -> out[i]
            // known as the delta rule: -(target - out) * derivative(out) * out 
            for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                if (layers[layerIndex].ActivationType == LayerType.Dropout) continue;
                if (layers[layerIndex].ActivationType == LayerType.Softmax) continue;
                layers[layerIndex].Delta.Zero();
 
                for (int sampleIndex = 0; sampleIndex < trainingSet.SampleCount; sampleIndex++)
                {
                    int shuffledIndex = trainingSet.ShuffledSample(sampleIndex).Index;

                    Span<float> a = layerIndex == layers.Length - 1 ? actual.Row(populationIndex, shuffledIndex) : layers[layerIndex].Neurons.AsSpan();
                    Span<float> e = layerIndex == layers.Length - 1 ? expected.Row(shuffledIndex) : layers[layerIndex].Gamma; //gamma[layerIndex];

                    // calc derivate if actual output == gamma 
                    if (layers[layerIndex].ActivationType == LayerType.Softmax)
                    {
                        // softmax with crossentropy reduces to a simple substraction
                        Intrinsics.Substract(e, a, layers[layerIndex].Gamma);
                    }
                    else
                    {
                        layers[layerIndex].Derivate(a, layers[layerIndex].Gamma);
                    }

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
                if (layers[layerIndex] is ParameterLayer currentLayer)
                {
                    int previousLayerIndex = layerIndex - 1;
                    if (layers[previousLayerIndex].ActivationType == LayerType.Dropout)
                    {
                        previousLayerIndex--;
                    }

                    // update biases 
                    for (int j = 0; j < layers[layerIndex].Size; j++)
                    {
                        float delta = layers[layerIndex].Gamma[j] * biasLearningRate;
                        currentLayer.Biases[j] -= delta + (currentLayer.BiasDeltas[j] * momentum);
                        currentLayer.BiasDeltas[j] = delta;
                    }

                    // update weights from error 
                    for (int i = 0; i < layers[layerIndex].Size; i++)
                    {
                        for (int j = 0; j < layers[previousLayerIndex].Size; j++)
                        {
                            float wDelta = weightLearningRate * layers[layerIndex].Delta[i] * layers[layerIndex - 1].Gamma[j];

                            currentLayer.Weights[i, j] -=
                                // current weight to error delta
                                wDelta
                                // momentum 
                                + (currentLayer.WeightDeltas![i, j] * momentum)
                                // strengthen learned 
                                + (weightLearningRate * (layers[previousLayerIndex].Gamma[j] - weightCost * currentLayer.Weights[i, j]));

                            currentLayer.WeightDeltas[i, j] = wDelta;
                        }
                    }

                }
            });

            Cost = Math.Abs(totalError);
        }

        /// <summary>
        /// backtrack state through derivates of the activation minimizing cost/error
        /// </summary>                
        public void BackPropagateOnline(
            SampleSet samples,
            int sampleIndex, 
            float[][] gamma, 
            float weightLearningRate = 0.01f, float biasLearningRate = 0.01f, float momentum = 1f, float weightCost = 0.00001f, float minCostDelta = 0.0000001f)
        {
            Span<float> inputs = samples.ShuffledData(sampleIndex);
            Span<float> expected = samples.ShuffledExpectation(sampleIndex);
       
            // initialize neurons from input patterns
            FeedForward(inputs);

            // calculate delta 2 output 
            float[] delta = new float[Output.Neurons.Length];
            float cost;

            if (Output.ActivationType == LayerType.Softmax)
            {
                cost = CalculateCrossEntropyDeltaAndCost(Output.Neurons, expected, delta); 
            }
            else
            {
                CalculateDelta(Output.Neurons, expected, delta);
                cost = CalculateCost(delta);
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
                if (layers[i].ActivationType == LayerType.Dropout) continue;

                int nextLayerIndex = i + 1;  
                if(layers[nextLayerIndex].ActivationType == LayerType.Dropout)
                {
                    nextLayerIndex++;
                }
                int previousLayerIndex = i - 1;
                if (layers[previousLayerIndex].ActivationType == LayerType.Dropout)
                {
                    previousLayerIndex--;
                }

                // update gamma from layer weights and current gamma on output
                for (int j = 0; j < layers[i].Size; j++)
                {
                    int k = 0;

                    if (layers[nextLayerIndex] is ParameterLayer nextLayer)
                    {
                        gamma[i][j] = 0;
                        // 
                        //  get weighed sum of gamma * previous neurons 
                        // 
                        if (Avx.IsSupported && gamma[nextLayerIndex].Length > 15)
                        {
                            Span<Vector256<float>> g = MemoryMarshal.Cast<float, Vector256<float>>(gamma[nextLayerIndex]);
                            Vector256<float> g0 = Vector256<float>.Zero;
                            int l = 0;
                            while (k < (gamma[nextLayerIndex].Length & ~7))
                            {
                                g0 = Intrinsics.MultiplyAdd(g[l], Vector256.Create
                                    (
                                        // could use gather.. 
                                        nextLayer.Weights[k + 0, j],
                                        nextLayer.Weights[k + 1, j],
                                        nextLayer.Weights[k + 2, j],
                                        nextLayer.Weights[k + 3, j],
                                        nextLayer.Weights[k + 4, j],
                                        nextLayer.Weights[k + 5, j],
                                        nextLayer.Weights[k + 6, j],
                                        nextLayer.Weights[k + 7, j]
                                    ), g0);
                                k += 8;
                                l++;
                            }
                            gamma[i][j] = Intrinsics.HorizontalSum(g0);
                        }
                        while (k < gamma[nextLayerIndex].Length)
                        {
                            gamma[i][j] += gamma[nextLayerIndex][k] * nextLayer.Weights[k, j];
                            k++;
                        }
                    }
                    else
                    {
                        // SOFTMAX == BROKEN
                        // non weighted layer, ie: softmax 
                        gamma[i][j] = Intrinsics.Sum(gamma[nextLayerIndex]);
                    }

                    //
                    // Calculate the new gamma from the activation derivate
                    //
                    layers[i].CalculateGamma(gamma[i], gamma[i], layers[i].Neurons);
                }

                // update layer weights and biases
                layers[i].Update(layers[previousLayerIndex], gamma[i], weightLearningRate, biasLearningRate, momentum, weightCost); 
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
                if (layers[i] is ParameterLayer layer)
                {
                    bool ownRandom = false;
                    if (random == null)
                    {
                        ownRandom = true;
                        random = new UniformRandom(0, 1f);
                    }
                    if (layer.Biases != null)
                    {
                        Span<float> rnd = random.Span(layer.Biases.Length);

                        for (int j = 0; j < layer.Biases.Length; j++)
                        {
                            bool b = rnd[j] > chance; // random.NextSingle() > chance; 

                            layer.Biases[j] =
                                b
                                ?
                                layer.Biases[j] += (biasRange + biasRange) * rnd[rnd.Length - j - 1] - biasRange    //rnd(-biasRange, biasRange)
                                :
                                layer.Biases[j];

                            mutationCount += b ? 1 : 0;  // (int)b  
                        }
                    }
                    //}
                    //for (int i = 1; i < LayerCount; i++)
                    //{
                    if (layer.Weights != null)
                    {
                        for (int j = 0; j < layer.Weights.GetLength(0); j++)
                        {
                            Span<float> rnd = random.Span(layer.Weights.GetLength(1));
                            for (int k = 0; k < layer.Weights.GetLength(1); k++)
                            {
                                bool b = rnd[k] > chance;

                                layer.Weights[j, k] =
                                    b
                                    ?
                                    layer.Weights[j, k] += (weightRange + weightRange) * rnd[rnd.Length - k - 1] - weightRange
                                    :
                                    layer.Weights[j, k];

                                mutationCount += b ? 1 : 0;
                            }
                        }
                    }
                    if (ownRandom)
                    {
                        random.Dispose();
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
                if (into.layers[i] is ParameterLayer intoLayer)
                {
                    for (int j = 0; j < layers[i].Neurons.Length; j++)
                    {
                        intoLayer.Neurons[j] = layers[i].Neurons[j];
                    }

                    if (layers[i] is ParameterLayer layer)
                    {
                        for (int j = 0; j < layer.Biases.Length; j++)
                        {
                            intoLayer.Biases[j] = layer.Biases[j];
                        }
                        for (int j = 0; j < layer.Weights.GetLength(0); j++)
                        {
                            for (int k = 0; k < layer.Weights.GetLength(1); k++)
                            {
                                intoLayer.Weights[j, k] = layer.Weights[j, k];
                            }
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
                if (layer.HasParameters)
                {
                    c +=
                        layer.Size * 1 //  bias
                        +
                        layer.Size * layer.PreviousSize * 1; // weights
                }
                else
                {
                    //c += layer.Size; 
                }
            }

            return c; 
        }
    }
}

