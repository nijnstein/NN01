using NSS;

namespace NN01
{
    public class NeuralNetworkBuilder
    {
        internal List<Layer> stack;

        protected NeuralNetworkBuilder()
        {
            stack = new List<Layer>();
        }

        static public NeuralNetworkBuilder FromInput(int inputLayerSize)
        {
            NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
            builder.stack.Add(new InputLayer(inputLayerSize));
            return builder;
        }
        static public NeuralNetworkBuilder FromInput2D(Size2D size)
        {
            NeuralNetworkBuilder builder = new NeuralNetworkBuilder();
            builder.stack.Add(new Input2D(size, Size2D.Zero));
            return builder;
        }
    }
}

