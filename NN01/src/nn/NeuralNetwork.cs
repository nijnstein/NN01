using ILGPU;
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

            if (random == null) random = new CPURandom(RandomDistributionInfo.Default); 

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

        public float CalculateError(float ideal, float actual) => .5f * (ideal - actual).Square(); 
        public float CalculateError(Span<float> ideal, Span<float> actual) => 0.5f * Intrinsics.SumSquaredDifferences(ideal, actual); 
        public float CalculateActivationDelta(float ideal, float actual) => -(ideal - actual);

 
        /// <summary>
        /// 
        ///  Feed forward all samples in the set  
        ///  
        ///  - uses shuffled indices 
        ///  - updates sample error 
        ///  - stored actual|output activation in samples.Actual buffer instead of network output neurons
        /// 
        /// </summary>
        /// <returns>total input set error</returns>
        public float FeedForward(SampleSet samples, Accelerator? acc)
        {
            float error = 0f;

            // reset delta & gamma buffers 
            for (int i = 0; i < layers.Length - 1; i++)
            {
                layers[i].Delta.Zero();
                layers[i].Gamma.Zero(); 
            }

            if (!gpu_feedforward(acc, samples, out error))
            {
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

                    // apply softmax normalization on the last layer (actual) if enabled 
                    if (layers[layers.Length - 1].Softmax)
                    {
                        MathEx.Softmax(actual, actual, true);
                    }

                    // calc error on sample
                    float sampleError = CalculateError(ideal, actual);

                    error += sampleError;
                    samples.SetShuffledError(sampleIndex, sampleError);
                }
            }
            
            return error; 
        }
                
        /// <summary>
        /// Specialized FeedForward GPU Kernel 
        /// - 4 layers 
        /// - activations: ReLU -> TanH -> LeakyRelu 
        /// - cost: meansquared 
        /// </summary>
        private static void gpu_feedforward_4_relu_tanh_lerelu_batch(
            Index1D sampleIndex,
            ArrayView2D<float, Stride2D.DenseX> data,
            ArrayView2D<float, Stride2D.DenseX> expectation,
            ArrayView1D<float, Stride1D.Dense> sampleErrors,
            ArrayView2D<float, Stride2D.DenseX> w0,
            ArrayView1D<float, Stride1D.Dense> b0,
            ArrayView1D<float, Stride1D.Dense> h0,
            ArrayView2D<float, Stride2D.DenseX> w1,
            ArrayView1D<float, Stride1D.Dense> b1,
            ArrayView1D<float, Stride1D.Dense> h1,
            ArrayView2D<float, Stride2D.DenseX> w2,
            ArrayView1D<float, Stride1D.Dense> b2,
            ArrayView2D<float, Stride2D.DenseX> actual,
            ArrayView1D<float, Stride1D.Dense> h2)
        {
            int sampleSize = w0.IntExtent.Y;
            int classCount = h2.IntLength;

            // Interop.Write("sample size: {0}, classes: {1} ", sampleSize, classCount);

            // ReLU 0 -> 1 
            //layers[1].Activate(layers[0], sample, layers[1].Neurons);
            gpu_relu(sampleIndex, sampleSize, sampleSize * sampleIndex, data.BaseView, h0, w0, b0);

            // TanH 1 -> 2
            //layers[2].Activate(layers[1]);
            gpu_relu(sampleIndex, sampleSize, 0, h0, h1, w1, b1);

            // LeakyReLU 2 -> 3 
            //layers[3].Activate(layers[2], layers[layers.Length - 2].Neurons, actual);
            gpu_relu(sampleIndex, sampleSize, 0, h1, h2, w2, b2);

            // calc error on sample expectation 
            // int l = info.ClassCount;
            float sum = 0; 
            for (int i = 0; i < classCount; i++)
            {
                float f = (expectation[sampleIndex, i] - h2[i]);
                actual[sampleIndex, i] = h2[i];
                sum += f * f; 
            }
            sampleErrors[sampleIndex] = sum * 0.5f;            
        }

        static void gpu_relu(int sampleIndex, int sampleSize, int baseIndex, ArrayView<float> n0, ArrayView<float> n1, ArrayView2D<float, Stride2D.DenseX> w01, ArrayView<float> b1)
        {
            //Interop.Write("-{0}-", n0.Length);       
            for (int index = 0; index < n1.IntLength; index++)
            {
                // index 0...j foreach [] in n01 
                float sum = 0f;

                // weighted sum 
                for (int i0 = 0; i0 < w01.Extent.Y; i0++)
                {
                    sum += w01[index, i0] * n0[baseIndex + i0];
                    /// sum += w01[index, i0] * n0[baseIndex + i0];
                }

                sum += b1[index];

                // ReLU
                n1[index] = MathF.Max(sum, 0);
            }
        }




        private static Action<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>>? get_gpu_feedforward_kernel(int layerCount, LayerActivationFunction a1, LayerActivationFunction a2, LayerActivationFunction a3, bool batched, bool softmax)
        {
            switch (layerCount)
            {

                case 4:
                    {
                        if( a1 == LayerActivationFunction.ReLU && a2 == LayerActivationFunction.Tanh && a3 == LayerActivationFunction.LeakyReLU)
                        {
                            if (softmax)
                            {
                                if (batched)
                                {
                                }
                                else
                                {
                                }
                            }
                            else
                            {
                                if (batched)
                                {
                                    return gpu_feedforward_4_relu_tanh_lerelu_batch;
                                }
                                else
                                {

                                }
                            }                            
                        }
                    }
                    break; 

            }
            return null; 
        }


        /// <summary>
        /// returns true if a kernel could be found matching current topology and data if fed forward
        /// </summary>
        /// <param name="acc">gpu accelerator</param>
        private bool gpu_feedforward(Accelerator? acc, SampleSet samples, out float error)
        {
            if(acc == null)
            {
                error = 1; 
                return false; 
            }

            Debug.Assert(samples != null);

            // get a kernel for given topology 
            var kernelObject = get_gpu_feedforward_kernel(4, LayerActivationFunction.ReLU, LayerActivationFunction.Tanh, LayerActivationFunction.LeakyReLU, true, false);
            if(kernelObject == null)
            {
                error = 0; 
                return false;
            }

            using var w0 = acc.Allocate2DDenseX<float>(layers[1].Weights);
            using var b0 = acc.Allocate1D<float>(layers[1].Biases);
            using var h0 = acc.Allocate1D<float>(layers[1].Size);

            using var w1 = acc.Allocate2DDenseX<float>(layers[2].Weights);
            using var b1 = acc.Allocate1D<float>(layers[2].Biases);
            using var h1 = acc.Allocate1D<float>(layers[2].Size);

            using var w2 = acc.Allocate2DDenseX<float>(layers[3].Weights);
            using var b2 = acc.Allocate1D<float>(layers[3].Biases);
            using var h2 = acc.Allocate1D<float>(layers[3].Size);

            // every 4 layer network has the same kernel format (can we use delegates to kernels?)
            acc.LaunchAutoGrouped(
                kernelObject,
                acc.DefaultStream,
                new Index1D(samples.SampleCount),
                samples.gpu_data.View, samples.gpu_expectation.View, samples.gpu_sampleErrors.View,
                w0.View, b0.View, h0.View,
                w1.View, b1.View, h1.View,
                w2.View, b2.View, samples.gpu_actual.View, h2.View);

            // now calculate from that state the error 
            // -> this is only safe when using a single thread on a single population
            //
            //  the error and actual buffer are written too from multiple threads -> watch out!
            //

            // sum the error on the gpu, less data to return over memory 
            // - returns 0.... 
            //float[] sum = new float[1] { 0 };
            //using var sums = acc.Allocate1D<float>(sum); 
            //acc.LaunchAutoGrouped<Index1D, ArrayView<float>, ArrayView<float>>(
            //    (index, errors, sum) => sum[0] += errors[index],
            //    acc.DefaultStream,
            //    samples.SampleCount, 
            //    samples.gpu_sampleErrors.View, 
            //    sums.View
            //);
            //
            //acc.Synchronize();
            //sums.CopyToCPU(sum);
            //error = sum[0]; 

            samples.gpu_sampleErrors.View.CopyToCPU(samples.sampleError);
            error = Intrinsics.Sum(samples.sampleError);

            return true; 
        }



        /// <summary>
        /// Backpropagate a full training set 
        /// </summary>
        /// <param name="trainingSet"></param>
        /// <param name="gamma"></param>
        /// <param name="weightLearningRate"></param>
        /// <param name="biasLearningRate"></param>
        /// <param name="momentum"></param>
        /// <param name="weightCost"></param>
        /// <param name="minCostDelta"></param>
        public void BackPropagateBatch(
            SampleSet trainingSet,
            float weightLearningRate = 0.01f, float biasLearningRate = 0.01f, float momentum = 1f, float weightCost = 0.00001f, float minCostDelta = 0.0000001f,
            IRandom random = null,
            Accelerator acc = null)
        {
            // feed forward all samples and get the total error 
            float totalError = 1 - (FeedForward(trainingSet, acc) / (trainingSet.SampleSize * trainingSet.SampleCount * trainingSet.Variance));

            // calculate total error change foreach output
            Span2D<float> actual = trainingSet.Actual.AsSpan2D<float>();
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
                for (int sampleIndex = 0; sampleIndex < trainingSet.SampleCount; sampleIndex++)
                {
                    int shuffledIndex = trainingSet.ShuffledSample(sampleIndex).Index;

                    Span<float> a = layerIndex == layers.Length - 1 ? actual.Row(shuffledIndex) : layers[layerIndex].Neurons.AsSpan();
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
                }
                Intrinsics.MultiplyScalar(layers[layerIndex].Delta, 1f / layers[layerIndex].Size); 
            }

            for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                layers[layerIndex - 1].Gamma.Zero();
                // calc hidden activation error 
                for (int j = 0; j < layers[layerIndex - 1].Size; j++)
                {
                    for (int i = 0; i < layers[layerIndex].Size; i++)
                    {
                        layers[layerIndex - 1].Gamma[j] += layers[layerIndex].Delta[i] * layers[layerIndex].Weights[i, j];
                    }
                }
            }

            for (int layerIndex = layers.Length - 1; layerIndex > 0; layerIndex--)
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
                            + (layers[layerIndex].WeightDeltas![i][j] * momentum)
                            // strengthen learned 
                            + (weightLearningRate * (layers[layerIndex - 1].Gamma[j] - weightCost * layers[layerIndex].Weights[i, j]));


                        layers[layerIndex].WeightDeltas[i][j] = wDelta; 
                    }
                }
            }

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

            for (int i = 1; i < LayerCount; i++)
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
            }

            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < layers[i].Weights.GetLength(0); j++)
                {
                    Span<float> rnd = random.Span(layers[i].Weights.GetLength(1));
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
            }

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

