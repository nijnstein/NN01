using ILGPU;
using ILGPU.Runtime;
using NSS.Neural;
using NSS;
using System.Diagnostics;
using System.Drawing;
using System.ComponentModel;
using System;
using System.Runtime.CompilerServices;
using System.Drawing.Printing;

namespace NN01
{
    public partial class NeuralNetwork
    {

        static void gpu_multiply_scalar(Index2D index, ArrayView2D<float, Stride2D.DenseX> a, float b)
            => a[index.X, index.Y] *= b;

        // sample[pop, sample, data]
        // neuron[pop, data]
        static void gpu_relu_pop(
            Index3D index,
            ArrayView2D<float, Stride2D.DenseX> n0,
            ArrayView2D<float, Stride2D.DenseX> n1,
            ArrayView3D<float, Stride3D.DenseXY> w01,
            ArrayView2D<float, Stride2D.DenseX> b1)
        {
            int size = w01.IntExtent.Z; 
            float sum = 0f;

            // weighted sum of state 
            for (int i1 = 0; i1 < size; i1++)
            {
                sum += w01[index.X, index.Z, i1] * n0[index.X, i1];
            }

            // plus bias 
            sum += b1[index.X, index.Z];

            // ReLU
            n1[index.X, index.Z] = MathF.Max(sum, 0);
        }

        // feed forward a sample from a single population
        static void gpu_feed_pop( 
            Index2D index, // population, sample
            nn3 nn,
            ArrayView2D<float, Stride2D.DenseX> sample,
            ArrayView3D<float, Stride3D.DenseXY> actual)
        {
            int populationIndex = index.X;
            int sampleIndex = index.Y;


            // l0 | Sample input 
            var size = nn.size0;
            var n1 = nn.n0;
            var w = nn.w0;
            var b = nn.b0;

            for (int i0 = 0; i0 < size; i0++)
            {
                float sum = 0f;
                // - weighted sum of state 
                for (int i1 = 0; i1 < w.IntExtent.Z; i1++)
                {
                    sum += w[populationIndex, i0, i1] * sample[sampleIndex, i1];
                }
                // - plus bias 
                sum += b[populationIndex, i0];
                // - ReLU
                n1[populationIndex, sampleIndex, i0] = MathF.Max(sum, 0);
            }

            // l2 
            size = nn.size1;
            var n0 = n1; 
            n1 = nn.n1;
            w = nn.w1;
            b = nn.b1;

            for (int i0 = 0; i0 < size; i0++)
            {
                float sum = 0f;
                for (int i1 = 0; i1 < w.IntExtent.Z; i1++)
                {
                    sum += w[populationIndex, i0, i1] * n0[populationIndex, sampleIndex, i1];
                }
                sum += b[populationIndex, i0];
                n1[populationIndex, sampleIndex, i0] = MathF.Max(sum, 0);
            }

            // l3 -> output into actual buffer 
            size = nn.size2;
            n0 = n1;
            n1 = nn.n2;
            w = nn.w2;
            b = nn.b2;

            for (int i0 = 0; i0 < size; i0++)
            {
                float sum = 0f;
                for (int i1 = 0; i1 < w.IntExtent.Z; i1++)
                {
                    sum += w[populationIndex, i0, i1] * n0[populationIndex, sampleIndex, i1];
                }
                sum += b[populationIndex, i0];

                actual[populationIndex, sampleIndex, i0] = MathF.Max(sum, 0);
            }
        }



        static void gpu_relu_derivate(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> input,
            ArrayView2D<float, Stride2D.DenseX> gamma)
        {
            gamma[index.X, index.Y] = input[index.X, index.Y] < 0 ? 0 : 1;
        }

        static void gpu_delta (
            Index3D index, // [population, sample, layersize]
            ArrayView2D<float, Stride2D.DenseX> expected,
            ArrayView3D<float, Stride3D.DenseXY> actual,
            ArrayView2D<float, Stride2D.DenseX> delta,
            ArrayView2D<float, Stride2D.DenseX> gamma
            )
        {
            // calc error delta with respect to derivate of actual output 
            // float inverse = 1 / (trainingSet.SampleCount * a.Length);

            delta[index.X, index.Z] +=
                gamma[index.X, index.Z]
                *
                // expected is sampledata indexed with sample index
                -(expected[index.Y, index.Z] - actual[index.X, index.Y, index.Z]);
        }

        static void gpu_delta_from_gamma(
     Index3D index, // [population, sample, layersize]
     ArrayView2D<float, Stride2D.DenseX> expected,
     ArrayView3D<float, Stride3D.DenseXY> actual,
     ArrayView2D<float, Stride2D.DenseX> delta,
     ArrayView2D<float, Stride2D.DenseX> gamma
     )
        {
            // calc error delta with respect to derivate of actual output 
            // float inverse = 1 / (trainingSet.SampleCount * a.Length);

            delta[index.X, index.Z] +=
                gamma[index.X, index.Z]
                *
                // expected is gamma of previous indexed with population index 
                -(expected[index.X, index.Z] - actual[index.X, index.Y, index.Z]);
        }

        static void gpu_delta(
            Index3D index, // [population, sample, layersize]
            ArrayView2D<float, Stride2D.DenseX> expected,
            ArrayView2D<float, Stride2D.DenseX> actual,
            ArrayView2D<float, Stride2D.DenseX> delta,
            ArrayView2D<float, Stride2D.DenseX> gamma
            )
        {
            // calc error delta with respect to derivate of actual output 
            // float inverse = 1 / (trainingSet.SampleCount * a.Length);

            delta[index.X, index.Z] +=
                gamma[index.X, index.Z]
                *
                // expected == gamma from previous foreach population 
                -(expected[index.X, index.Z] - actual[index.X, index.Z]);
        }


        static void gpu_gamma(
              Index3D index,//nn.populationcount, nn.size1, nn.size2 
              ArrayView2D<float, Stride2D.DenseX> gamma0,
              ArrayView2D<float, Stride2D.DenseX> delta1,
              ArrayView3D<float, Stride3D.DenseXY> w1)
        {
            gamma0[index.X, index.Y] = gamma0[index.X, index.Y] + (delta1[index.X, index.Z] * w1[index.X, index.Z, index.Y]); 
            //layers[layerIndex - 1].Gamma[j] += layers[layerIndex].Delta[i] * layers[layerIndex].Weights[i, j];
        }

     
        /// <summary>
        /// feed forward all samples
        /// </summary>
        public static void gpu_batched_population_feedforward_error(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> expectations,
            ArrayView3D<float, Stride3D.DenseXY> actual,
            ArrayView2D<float, Stride2D.DenseX> error,
            nn3 data)
        {
            int populationIndex = index.X;
            int sampleIndex = index.Y;

            int sampleSize = data.samplesize;
            int classCount = data.classcount;

            // - calc error on sample expectation 
            // - copy actual activation for sample 
            float sum = 0;
            for (int classIndex = 0; classIndex < classCount; classIndex++)
            {
                float f = actual[populationIndex, sampleIndex, classIndex] - expectations[sampleIndex, classIndex];
                // = data.n2[populationIndex, classIndex];
                sum += f * f;
            }
            error[populationIndex, sampleIndex] = sum * 0.5f;
        }

        static void gpu_zero(Index2D index, ArrayView2D<float, Stride2D.DenseX> a) => a[index.X, index.Y] = 0f;

        static float gpu_feedforward(Accelerator? acc, nn3 nn, SampleSet samples)
        {
            // Console.WriteLine($"{nn.populationcount} {nn.size0} {nn.size1} {nn.size2}");
            // zero out delta and gamma buffers
            acc.LaunchAutoGrouped(gpu_zero, new Index2D(nn.populationcount, nn.size0), nn.d0);
            acc.LaunchAutoGrouped(gpu_zero, new Index2D(nn.populationcount, nn.size1), nn.d1);
            acc.LaunchAutoGrouped(gpu_zero, new Index2D(nn.populationcount, nn.size2), nn.d2);
            acc.LaunchAutoGrouped(gpu_zero, new Index2D(nn.populationcount, nn.samplesize), nn.g0);
            acc.LaunchAutoGrouped(gpu_zero, new Index2D(nn.populationcount, nn.size0), nn.g1);
            acc.LaunchAutoGrouped(gpu_zero, new Index2D(nn.populationcount, nn.size1), nn.g2);

            // feed all samples through from all populations
            acc.LaunchAutoGrouped(
                gpu_feed_pop, 
                new Index2D(nn.populationcount, nn.samplecount), 
                nn, 
                samples.gpu_data.View, 
                samples.gpu_actual.View);

            //acc.LaunchAutoGrouped(gpu_relu_pop, new Index3D(nn.populationcount, nn.samplecount, nn.size0), samples.gpu_data.View, nn.n0, nn.w0, nn.b0);
            //acc.LaunchAutoGrouped(gpu_relu_pop, new Index3D(nn.populationcount, nn.samplecount, nn.size1), nn.n0, nn.n1, nn.w1, nn.b1);
            //acc.LaunchAutoGrouped(gpu_relu_pop, new Index3D(nn.populationcount, nn.samplecount, nn.size2), nn.n1, nn.n2, nn.w2, nn.b2);

            // calculate the error for each sample
            acc.LaunchAutoGrouped(gpu_batched_population_feedforward_error,
                acc.DefaultStream,
                new Index2D(nn.populationcount, nn.samplecount),
                  samples.gpu_expectation.View,
                  samples.gpu_actual.View,
                  samples.gpu_sampleErrors.View,
                  nn);

            samples.gpu_sampleErrors.CopyToCPU(samples.sampleError);
            acc.Synchronize();

            float cost = Intrinsics.Average(samples.sampleError.AsSpan<float>()) / nn.populationcount;// * nn.samplecount);
            return cost;
        }

        static void gpu_fitness(Accelerator? acc, nn3 nn, SampleSet samples)
        {

        }

        static void gpu_backprop(Accelerator? acc, nn3 nn, SampleSet samples)
        {
            // foreach  population:     calculate last layer gamma, then delta 
            // foreach  population:     >  derivate(actual) -> gamma2
            // foreach  population:     >  calc delta to 2
            // foreach  population:     >  calc gamma 1 to 2 

            // - output gamma from batch is calculated from averaged actual 
            acc.LaunchAutoGrouped<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>>(
                (index, a, avg) =>
                {
                    avg[index.X, index.Z] += a[index.X, index.Y, index.Z];
                },
                new Index3D(nn.populationcount, nn.samplecount, nn.size2),
                samples.gpu_actual.View,
                nn.g2);

            acc.LaunchAutoGrouped(gpu_multiply_scalar, new Index2D(nn.populationcount, nn.size2), nn.g2, 1f / nn.size2);
            acc.LaunchAutoGrouped(gpu_relu_derivate, new Index2D(nn.populationcount, nn.size2), nn.g2, nn.g2);

            acc.LaunchAutoGrouped(gpu_delta,
                new Index3D(nn.populationcount, nn.samplecount, nn.size2),
                samples.gpu_expectation.View,
                samples.gpu_actual.View, // pop smpl size
                nn.d2,
                nn.g2);


            acc.LaunchAutoGrouped(gpu_multiply_scalar, new Index2D(nn.populationcount, nn.size2), nn.d2, 1f / (nn.samplecount * nn.size2.Square()));
            acc.LaunchAutoGrouped(gpu_gamma, new Index3D(nn.populationcount, nn.size1, nn.size2), nn.g1, nn.d2, nn.w2);


            // foreach  population:     calculate nxt layer gamma, then delta 
            // foreach  population:     >  derivate(gamma2) -> gamma1
            // foreach  population:     >  calc delta to 1
            // foreach  population:     >  calc gamma 0 to 1 
            acc.LaunchAutoGrouped(gpu_relu_derivate, new Index2D(nn.populationcount, nn.size1), nn.g1, nn.g1);
            acc.LaunchAutoGrouped(gpu_delta_from_gamma,
                new Index3D(nn.populationcount, nn.samplecount, nn.size1),
                nn.g2,
                nn.n1,
                nn.d1,
                nn.g1);
            acc.LaunchAutoGrouped(gpu_multiply_scalar, new Index2D(nn.populationcount, nn.size1), nn.d1, 1f / (nn.samplecount * nn.size1.Square()));
            acc.LaunchAutoGrouped(gpu_gamma, new Index3D(nn.populationcount, nn.size0, nn.size1), nn.g0, nn.d1, nn.w1);

            // foreach  population:     calculate lst layer gamma, then delta 
            // foreach  population:     >  derivate(gamma1) -> gamma0
            // foreach  population:     >  calc delta to 1
            // foreach  population:     >  calc gamma 0 to 1 
            acc.LaunchAutoGrouped(gpu_relu_derivate, new Index2D(nn.populationcount, nn.size0), nn.g0, nn.g0);
            acc.LaunchAutoGrouped(gpu_delta_from_gamma,
                new Index3D(nn.populationcount, nn.samplecount, nn.size0),
                nn.g1,
                nn.n0,
                nn.d0,
                nn.g0);
            acc.LaunchAutoGrouped(gpu_multiply_scalar, new Index2D(nn.populationcount, nn.size0), nn.d0, 1f / (nn.samplecount * nn.size0.Square()));
            acc.LaunchAutoGrouped(gpu_gamma, new Index3D(nn.populationcount, nn.samplesize, nn.size0), nn.g0, nn.d0, nn.w0);
            acc.Synchronize();
        }

        public void gpu_update_layers(Accelerator? acc, nn3 nn, SampleSet samples,
            float weightLearningRate, 
            float weightCost,
            float biasLearningRate,
            float momentum)
        { 
            // calculate new weights and biases for all layers 
            
            // l2
            acc.LaunchAutoGrouped(gpu_update_bias, new Index2D(nn.populationcount, nn.size2), 
                nn.b2, nn.db2, nn.g2, biasLearningRate, momentum); 
            
            acc.LaunchAutoGrouped(gpu_update_weight, new Index3D(nn.populationcount, nn.size2, nn.size1), 
                nn.w2, nn.dw2, nn.d2, nn.g2, weightLearningRate, weightCost, momentum);

            // l1
            acc.LaunchAutoGrouped(gpu_update_bias, new Index2D(nn.populationcount, nn.size1),
                nn.b1, nn.db1, nn.g1, biasLearningRate, momentum);

            acc.LaunchAutoGrouped(gpu_update_weight, new Index3D(nn.populationcount, nn.size1, nn.size0),
                nn.w1, nn.dw1, nn.d1, nn.g1, weightLearningRate, weightCost, momentum);

            // l0
            acc.LaunchAutoGrouped(gpu_update_bias, new Index2D(nn.populationcount, nn.size0),
                nn.b0, nn.db0, nn.g0, biasLearningRate, momentum);

            acc.LaunchAutoGrouped(gpu_update_weight, new Index3D(nn.populationcount, nn.size0, nn.samplesize),
                nn.w0, nn.dw0, nn.d0, nn.g0, weightLearningRate, weightCost, momentum);
        }

        // pop - layersize
        static void gpu_update_bias(
            Index2D index,// [population, layersize]
            ArrayView2D<float, Stride2D.DenseX> bias,
            ArrayView2D<float, Stride2D.DenseX> biasDelta,
            ArrayView2D<float, Stride2D.DenseX> gamma, 
            float biasLearningRate, 
            float biasMomentum)
        {
            float delta = gamma[index.X, index.Y] * biasLearningRate;
            bias[index.X, index.Y] -= delta + (biasDelta![index.X, index.Y] * biasMomentum);
            biasDelta[index.X, index.Y] = delta;
        }

        static void gpu_update_weight(
            Index3D index, // [population, size, previoussize]
            ArrayView3D<float, Stride3D.DenseXY> weights,
            ArrayView3D<float, Stride3D.DenseXY> weightDeltas,
            ArrayView2D<float, Stride2D.DenseX> deltaActivation, // = old neurons
            ArrayView2D<float, Stride2D.DenseX> prevGamma,
            float weightLearningRate,
            float weightCost,
            float momentum)
        {
            int i = index.Y; // size 
            int j = index.Z; // previous 

            float delta = deltaActivation[index.X, i] * prevGamma[index.X, j] * weightLearningRate;

            weights[index.X, i, j] -=
                // delta 
                delta
                // momentum 
                + (weightDeltas![index.X, i, j] * momentum)
                // strengthen learned weights
                + (weightLearningRate * (prevGamma[index.X, j] - weightCost * weights[index.X, i, j]));

            weightDeltas[index.X, i, j] = delta;
        }
            

        internal void gpu_BackPropagateBatch(
            nn3 nn,
            int populationIndex,
            SampleSet trainingSet,
            float weightLearningRate = 0.01f, float biasLearningRate = 0.01f, float momentum = 1f, float weightCost = 0.00001f, float minCostDelta = 0.0000001f,
            IRandom random = null,
            Accelerator acc = null)
        {
            float totalError = 1 - gpu_feedforward(acc, nn, trainingSet);
              //  1 - (gpu_feedforward(acc, nn, trainingSet)
              // /
              //  (trainingSet.SampleSize * trainingSet.SampleCount * trainingSet.Variance));

            gpu_backprop(acc, nn, trainingSet); 

            gpu_update_layers(acc, nn, trainingSet, weightLearningRate, weightCost, biasLearningRate, momentum);

            this.Cost = Math.Abs(totalError); 
        }

        internal int gpu_MutatePopulation(nn3 nn3, Accelerator acc, int istep)
        {
            // foreach population member
            //  - copy parameters from a better network...
            //  - mutate those parameters
            return 0; 
        }
    }
}

