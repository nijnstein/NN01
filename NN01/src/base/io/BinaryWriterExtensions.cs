using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    static public class BinaryWriterExtensions
    {
        static public void Write(this BinaryWriter w, LayerActivationFunction activator)
        {
            w.Write((byte)activator);
        }
        static public void Write(this BinaryWriter w, LayerInitializationType initializer)
        {
            w.Write((byte)initializer);
        }
        static public void Write(this BinaryWriter w, Layer layer)
        {
            w.Write(layer.Size);
            w.Write(layer.PreviousSize);
            w.Write(layer.ActivationType);
            w.Write(layer.BiasInitializer);
            w.Write(layer.WeightInitializer);

            if (layer.PreviousSize > 0) // == IsInput 
            {
                for (int i = 0; i < layer.Size; i++)
                {
                    w.Write(layer.Biases[i]);
                }
                for (int i = 0; i < layer.Size; i++)
                {
                    for (int j = 0; j < layer.PreviousSize; j++)
                    {
                        w.Write(layer.Weights[i][j]);
                    }
                }
            }
        }

        static public void Write(this BinaryWriter w, NeuralNetwork nn)
        {
            w.Write("NN01");
            w.Write(nn.LayerCount);
            w.Write(nn.Fitness);
            w.Write(nn.Cost);
            for (int i = 0; i < nn.LayerCount; i++)
            {
                w.Write(nn[i]);
            }
        }
    }
}
