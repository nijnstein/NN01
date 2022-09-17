using NSS;

namespace NN01
{
    public class Dropout : Layer
    {
        public override LayerType ActivationType => LayerType.Dropout;
        public override LayerConnectedness Connectedness => LayerConnectedness.Full;
        
        public readonly float DropoutFactor;
        public readonly BitBuffer Dropouts; 

        public Dropout(int size, float dropoutFactor = 0.01f) : base(size, size) 
        {
            DropoutFactor = dropoutFactor;
            Dropouts = new BitBuffer(size);
            Dropouts.DisableAll(); 
        }

        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            if (inputData != outputData)
            {
                if (DropoutFactor > 0)
                {
                    int drops = 0; 
                    unchecked
                    {
                        for (int i = 0; i < inputData.Length; i++)
                        {
                            bool drop = Dropouts.IsSet(i); 
                            outputData[i] = drop ? 0 : inputData[i];
                            drops += drop ? 1 : 0; 
                        }
                    }
                    if(drops > 0)
                    {
                        // rescale output, accounting for the lost activation pressure dueue to drops
                        // =>   output / (1 - 1 / n * ndrop)
                        float scale = 1f - ((1f / inputData.Length) * drops);
                        Intrinsics.MultiplyScalar(outputData, 1f / scale, outputData); 
                    }
                }
                else
                {
                    inputData.CopyTo(outputData);
                }
            }
        }
      
        public override void Derivate(Span<float> input, Span<float> output)
        {
            if (input != output)
            {
                input.CopyTo(output);
            }
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            throw new NotImplementedException();
        }

        public void RandomizeDropout(IRandom random)
        {
            using(BernoulliRandom rng = new BernoulliRandom(DropoutFactor))
            {
                rng.Fill(Dropouts);
            }
        }
    }


}
