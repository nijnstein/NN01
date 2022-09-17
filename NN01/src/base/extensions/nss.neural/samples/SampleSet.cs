using ILGPU;
using ILGPU.Runtime;
using System.Data;
using System.Diagnostics;
using System.Security.Cryptography.X509Certificates;

namespace NSS.Neural
{
    public class SampleSet : IDisposable
    {
        readonly public int SampleSize;
        readonly public int SampleCount;
        readonly public int ClassCount;

        private Sample[] samples;
        private float[,] data;
        private float[,] expectation;
        private float[,,] actual;
        private int[] indexTable;
        public float[] probabilityIndex;
        public float[,] sampleError;

        internal MemoryBuffer2D<float, Stride2D.DenseX> gpu_data;
        internal MemoryBuffer2D<float, Stride2D.DenseX> gpu_expectation;
        internal MemoryBuffer3D<float, Stride3D.DenseXY> gpu_actual;
        internal MemoryBuffer2D<float, Stride2D.DenseX> gpu_sampleErrors;

        public float Variance;
        public float Average;

        public Sample[] Samples => samples;
        public float[,] Data => data;
        public float[,] Expectation => expectation;
        public float[,,] Actual => actual;
        public float[] ProbabilityIndex => probabilityIndex; 

        public Span<int> Indices => indexTable;
        public bool IsPrepared { get; private set; }
        public Span<float> SampleData(int sampleIndex) => data.AsSpan2D<float>().Row(sampleIndex);
        public Span<float> SampleExpectation(int sampleIndex) => expectation.AsSpan2D<float>().Row(sampleIndex);
        public float SampleError(int populationIndex, int sampleIndex) => sampleError[populationIndex, sampleIndex]; 
        public Sample ShuffledSample(int sampleIndex) => samples[indexTable[sampleIndex]]; 
        public Span<float> ShuffledData(int sampleIndex) => data.AsSpan2D<float>().Row(indexTable[sampleIndex]);
        public Span<float> ShuffledExpectation(int sampleIndex) => expectation.AsSpan2D<float>().Row(indexTable[sampleIndex]);
        public void SetSampleError(int populationIndex, int sampleIndex, float error) => sampleError[populationIndex, sampleIndex] = error;
        public void SetShuffledError(int populationIndex, int shuffledIndex, float error) => sampleError[populationIndex, indexTable[shuffledIndex]] = error;
        public Span<float> ShuffledActual(int populationIndex, int sampleIndex) => actual.AsSpan3D<float>().Row(populationIndex, indexTable[sampleIndex]);
        public Span<float> SampleActual(int populationIndex, int sampleIndex) => actual.AsSpan3D<float>().Row(populationIndex, sampleIndex);


        public bool IsGPUMemoryAllocated { get; protected set; } = false;


        public SampleSet(int sampleSize, int sampleCount, int classCount, int populationCount)
        {
            SampleSize = sampleSize;
            SampleCount = sampleCount;
            ClassCount = classCount;

            samples = new Sample[sampleCount];
            data = new float[sampleCount, sampleSize];
            expectation = new float[sampleCount, classCount];
            actual = new float[populationCount, sampleCount, classCount]; 
            sampleError = new float[populationCount, sampleCount]; 

            IsPrepared = false;

            gpu_data = null; 
            gpu_expectation = null;
            gpu_sampleErrors = null;
            gpu_actual = null;
        }

        public SampleSet(float[,] patterns, int[] classes, int classCount, bool softmax, int populationCount)
        {
            Debug.Assert(patterns != null && classes != null);

            SampleCount = patterns.GetLength(0);
            SampleSize = patterns.GetLength(1);
            ClassCount = classCount;

            Debug.Assert(SampleSize > 0);
            Debug.Assert(patterns.GetLength(0) == classes.GetLength(0));

            data = patterns;
            samples = new Sample[SampleCount];
            expectation = new float[SampleCount, ClassCount];
            actual = new float[populationCount, SampleCount, ClassCount];
            sampleError = new float[populationCount, SampleCount];

            for (int i = 0; i < SampleCount; i++)
            {
                samples[i] = new Sample(i, SampleSize, classes[i], ClassCount);
            }

            gpu_data = null;
            gpu_expectation = null;
            gpu_sampleErrors = null;
            gpu_actual = null;

            Prepare(softmax);
        }

        public void Prepare(bool softmax)
        { 
            GenerateExpectations(softmax);
            //CancelMeans();
            GenerateIndexArray();

            sampleError.Zero();
            actual.Zero(); 

            probabilityIndex = GetProbabilityIndex();
            Average = data.AsSpan<float>().Average();
            Variance = data.AsSpan<float>().Variance(Average);
            IsPrepared = true;     
        }


        public void GenerateExpectations(bool softmax)
        { 
            for (int i = 0; i < SampleCount; i++)
            {
                Span<float> e = expectation.AsSpan2D<float>().Row(i); 
                GenerateExpectation(i, e);
                if (softmax)
                {
                    // softmax has NO use if we use it on a ONE hot encoded expectation 
                    MathEx.Softmax(e, e, true);
                }
            }
        }

        public void Dispose()
        {
            data = null;
            expectation = null;
        }

        public void CancelMeans()
        {
            unchecked
            {
                for (int i = 0; i < SampleCount; i++)
                {
                    CancelMeans(i);
                }
            }
        }

        public void UpdateStatistics(int sampleIndex)
        {
            Span<float> sample = data.AsSpan2D<float>().Row(sampleIndex);

            samples[sampleIndex].Average = MathEx.Average(sample);
            samples[sampleIndex].Variance = MathEx.Variance(sample, samples[sampleIndex].Average);
        }

        public void CancelMeans(int sampleIndex)
        {
            Span<float> sampleData = data.AsSpan2D<float>().Row(sampleIndex);
            Sample sample = samples[sampleIndex];

            sample.Average = MathEx.Average(sampleData);

            float f = sample.Average / 2f;
            for (int i = 0; i < sampleData.Length; i++)
            {
                sampleData[i] -= f;
            }

            UpdateStatistics(sampleIndex);
        }


        public void GenerateExpectation(int sampleIndex, Span<float> target)
        {
            Debug.Assert(target.Length <= ClassCount);
            Debug.Assert(sampleIndex >= 0 && sampleIndex < SampleCount);

            int sampleClass = samples[sampleIndex].Class;

            switch (samples[sampleIndex].ExpectationEncoding)
            {
                case SampleExpectationEncodingType.OneHot:
                    unchecked
                    {
                        for (int i = 0; i < ClassCount; i++)
                        {
                            // == one hot encoding
                            target[i] = sampleClass == i + 1 ? 1f : 0f;
                        }
                    }
                    break;

                default:
                    throw new NotSupportedException($"sample index: {sampleIndex}, encodingtype: {samples[sampleIndex].ExpectationEncoding}");
            }
        }

        public Span<int> ShuffleIndices(Random random) => ShuffleIndices(indexTable, random);
        public Span<int> ShuffleIndices(IRandom random) => ShuffleIndices(indexTable, random);
        public Span<int> ShuffleIndices(Span<int> indices, Random random)
        {
            int n = indices.Length;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);

                int value = indices[k];
                indices[k] = indices[n];
                indices[n] = value;
            }
            return indices;
        }
        public Span<int> ShuffleIndices(Span<int> indices, IRandom random)
        {
            int n = indices.Length;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);

                int value = indices[k];
                indices[k] = indices[n];
                indices[n] = value;
            }
            return indices;
        }


        public void GenerateIndexArray()
        {
            indexTable = new int[SampleCount];
            GenerateIndexArray(indexTable); 
        }

        public Span<int> GenerateIndexArray(Span<int> indices)
        {
            unchecked
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    indices[i] = i;
                }
            }
            return indices; 
        }


        /// <summary>
        /// the probability of an index within the sample being set for all samples in the set
        /// </summary>
        /// <returns></returns>
        public float[] GetProbabilityIndex(float setThreshold = 0.05f)
        {
            // get probability of an input being set in this test batch 
            float[] pi = new float[SampleSize];
            MathEx.Zero(pi); 

            unchecked
            {
                for (int i = 0; i < SampleCount; i++)
                {
                    for(int j = 0; j < SampleSize; j++)
                    {
                        if (data[i, j] > setThreshold)
                        {
                            pi[j]++;
                        }
                    }
                }
                Intrinsics.MultiplyScalar(pi, 1f / SampleCount);       
            }

            return pi; 
        }


        public void AllocateAndCopyToGPU(Accelerator acc, int populationCount)
        {
            gpu_data = acc.Allocate2DDenseX<float>(new Index2D(Data.GetLength(0), Data.GetLength(1)));
            gpu_expectation = acc.Allocate2DDenseX<float>(new Index2D(Expectation.GetLength(0), Expectation.GetLength(1)));
            gpu_sampleErrors = acc.Allocate2DDenseX<float>(new Index2D(sampleError.GetLength(0), sampleError.GetLength(1)));
            gpu_actual = acc.Allocate3DDenseXY<float>(new Index3D(actual.GetLength(0), actual.GetLength(1), actual.GetLength(2))); 

            gpu_data.CopyFromCPU(data); 
            gpu_expectation.CopyFromCPU(expectation);

            acc.Synchronize(); 
        }

        public void ReleaseGPU(Accelerator acc)
        {
            gpu_data.Dispose();
            gpu_expectation.Dispose();
            gpu_sampleErrors.Dispose();
            gpu_actual.Dispose(); 
        }


    }
}