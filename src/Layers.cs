using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public enum LayerActivationFunction
    {
        Sigmoid, 

        Tanh, 
        
        /// <summary>
        /// Rectified Linear Unit 
        /// </summary>
        ReLU,

        /// <summary>
        /// Leaky Rectified Linear Unit
        /// </summary>
        LeakyReLU, 

        SoftMax,

        /// <summary>
        /// reduces to linear regression
        /// </summary>
        Linear, 
      
        /// <summary>
        /// swish by google 
        /// </summary>
        Swish,
        
        /// <summary>
        /// Binary Step
        /// </summary>
        Binary,

        None
    }

    public abstract class Layer
    {
        public float[] Neurons;
        public float[][] Weights;
        public float[] Biases;

        public int Size;
        public int PreviousSize;

        public bool IsInput => PreviousSize == 0;

        public abstract LayerActivationFunction ActivationType { get; }

        public Layer(int size, int previousSize)
        {
            Size = size;
            PreviousSize = previousSize;
            Neurons = new float[size];

            if (!IsInput)
            {
                Biases = new float[size];
                Weights = (IsInput ? null : new float[size][])!;
                for (int i = 0; i < size; i++)
                {
                    Biases[i] = Random.Shared.NextSingle() - 0.5f;
                    Weights![i] = new float[previousSize];
                    for (int j = 0; j < previousSize; j++)
                    {
                        Weights[i][j] = Random.Shared.NextSingle() - 0.5f;
                    }
                }
            }
            else
            {
                Biases = null!;
                Weights = null!;
            }
        }

        public abstract void Activate(Layer previous);
        public abstract void CalculateGamma(float[] delta, float[] gamma, float[] target);

        public void CalculateGamma(float[] gamma)
        {
            CalculateGamma(gamma, gamma, Neurons);
        }
    }

    public class BinaryLayer : Layer
    {       
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Binary;
        public BinaryLayer(int size, int previousSize) : base(size, previousSize) { }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j][k] * previous.Neurons[k];
                }

                Neurons[j] = value < 0 ? 0 : 1;
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)

            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * ((target[i] < 0) ? 0f : 1f);
            }
        }
    }

    public class ReLuLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.ReLU;
        public ReLuLayer(int size, int previousSize) : base(size, previousSize) { }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                // weighted sum of (w[j][k] * n[i][k])
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j][k] * previous.Neurons[k];
                }

                // relu 
                Neurons[j] = Math.Max(value, 0);
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)

            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * ((target[i] < 0) ? 0f : 1f);
            }
        }
    }
    public class LeakyReLuLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.LeakyReLU;
        public LeakyReLuLayer(int size, int previousSize) : base(size, previousSize) { }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                // weighted sum of (w[j][k] * n[i][k])
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j][k] * previous.Neurons[k];
                }

                // relu 
                Neurons[j] = value < 0 ? (float)Math.Exp(value) - 1 : value;
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * ((target[i] < 0) ? 0.01f : 1f);
            }
        }
    }
   
    public class TanhLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Tanh;
        public TanhLayer(int size, int previousSize) : base(size, previousSize) { }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                // weighted sum of (w[j][k] * n[i][k])
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j][k] * previous.Neurons[k];
                }
                Neurons[j] = (float)Math.Tanh(value);
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * (1 - (target[i] * target[i]));
            }
        }
    }
    public class SigmoidLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Sigmoid;
        public SigmoidLayer(int size, int previousSize) : base(size, previousSize) { }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                // weighted sum of (w[j][k] * n[i][k])
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j][k] * previous.Neurons[k];
                }

                // sigmoid activation  
                float f = (float)Math.Exp(value);
                Neurons[j] = f / (1.0f + f);
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * (target[i] * (1f - target[i]));
            }
        }
    }

    public class SwishLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Swish;
        public SwishLayer(int size, int previousSize) : base(size, previousSize) { }

        
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                // weighted sum of (w[j][k] * n[i][k])
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j][k] * previous.Neurons[k];
                }
                Neurons[j] = value.Swish();
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * target[i].SwishDerivative();
            }
        }
    }


    public class SoftMaxLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.SoftMax;
        public SoftMaxLayer(int size, int previousSize) : base(size, previousSize) { }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                Span<float> values = stackalloc float[previous.Size];

                for (int k = 0; k < previous.Size; k++)
                {
                    values[k] = Weights[j][k] * previous.Neurons[k];
                }

                MathExtensions.Softmax(values, Neurons);
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * (target[i] * (1f - target[i]));
            }
        }
    }

    public class LinearLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Linear;
        public LinearLayer(int size, int previousSize) : base(size, previousSize) { }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                for (int k = 0; k < previous.Size; k++)
                {
                    Neurons[k] = Weights[j][k] * previous.Neurons[k];
                }
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * target[i];
            }
        }
    }

    public class InputLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.None;
        public InputLayer(int size) : base(size, 0) { }

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
