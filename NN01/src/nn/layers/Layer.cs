using NSS;
using System.Diagnostics;
using System.Reflection;

namespace NN01
{

    public abstract class Layer
    {
        public int Size;
        public int PreviousSize;

        public float[] Neurons;
        public float[] Delta;   // error * derivate 
        public float[] Gamma;   // derivate of activation
        public bool IsInput => PreviousSize == 0;
        public abstract LayerType ActivationType { get; }
        public abstract LayerConnectedness Connectedness { get; }
        public virtual bool HasParameters { get; } = false;
        public Layer(int size, int previousSize)
        {
            Size = size;
            PreviousSize = previousSize;
            Neurons = new float[size];
            Gamma = new float[size];
            Delta = new float[size];
        }

        public abstract void Activate(Layer previous, Span<float> inputData, Span<float> outputData);
        public abstract void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target);
        public abstract void Derivate(Span<float> input, Span<float> output);
        
        public virtual void Activate(Layer previous)
        {
            Activate(previous, previous.Neurons, Neurons);
        }
        
        public virtual void Derivate(Span<float> output)
        {
            Derivate(Neurons, output);
        }
        
        public virtual void Update(Layer previous, float[] gamma, float weightLearningRate, float biasLearningRate, float momentum, float weightCost)
        {
            Debug.Assert(Neurons != null);
            Debug.Assert(previous != null && previous.Size == Size);
            Buffer.BlockCopy(previous.Neurons, 0, Neurons, 0, Neurons.Length);
        }

        public virtual void PassThrough(Layer previous)
        {
            Buffer.BlockCopy(previous.Neurons, 0, Neurons, 0, Neurons.Length);
        }

        /// <summary>
        /// generate different distributions from a uniform random number generator 
        /// uniform -1..0..1
        /// </summary>
        /// <param name="initializer"></param>
        /// <param name="data"></param>
        /// <param name="random"></param>
        protected void FillParameterDistribution(LayerInitializationType initializer, Span<float> data, IRandom random)
        {
            switch (initializer)
            {
                case LayerInitializationType.dot01: for (int i = 0; i < data.Length; i++) data[i] = 0.01f; break;
                case LayerInitializationType.Zeros: for (int i = 0; i < data.Length; i++) data[i] = 0; break;
                case LayerInitializationType.Ones: for (int i = 0; i < data.Length; i++) data[i] = 1; break;

                default:
                case LayerInitializationType.Random:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = MathF.Abs(random.NextSingle());
                        }
                    }
                    break;

                case LayerInitializationType.Normal:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = MathF.Abs(RandomDistributionInfo.NormalKernel(
                                random.NextSingle(), random.NextSingle(),
                                0, 1, 0));
                        }
                    }
                    break;

                case LayerInitializationType.Glorot:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = MathF.Abs(RandomDistributionInfo.GlorotKernel(
                                random.NextSingle(), random.NextSingle(),
                                0f, 1f, Size + PreviousSize));
                        }
                    }
                    break;

                case LayerInitializationType.HeNormal:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = MathF.Abs(RandomDistributionInfo.HeNormalKernel(
                                random.NextSingle(), random.NextSingle(),
                                0, 1, Size));
                        }
                    }
                    break;

                case LayerInitializationType.Xavier:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = RandomDistributionInfo.XavierKernel(random.NextSingle(), PreviousSize);
                        }
                    }
                    break;

                case LayerInitializationType.XavierNormalized:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = RandomDistributionInfo.XavierNormalizedKernel(random.NextSingle(), PreviousSize, Size);
                        }
                    }
                    break;

                case LayerInitializationType.Uniform:
                    {
                        MathEx.Uniform(data, 1f);
                    }
                    break;
            }
        }

    }
}
