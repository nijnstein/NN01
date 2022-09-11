using NUnit.Framework;
using NN01;
using System;
using BenchmarkDotNet.Attributes;
using System.Text;
using ILGPU.Algorithms.Random;
using NSS;
using NSS.GPU;
using System.Drawing;
using Microsoft.CodeAnalysis.Operations;
using Perfolizer.Mathematics.SignificanceTesting;
using Microsoft.Diagnostics.Tracing.Parsers.MicrosoftWindowsWPF;
using System.Diagnostics;
using System.Data.SqlTypes;
using Microsoft.Diagnostics.Tracing;

namespace UnitTests
{
    [TestFixture]
    public partial class UnitTest_Random
    {
        const int samplecount = 1000000;

        [TestCase()]
        public void RandomGPU_normal()
        {
            using GPURandom rnd = new GPURandom(RandomDistributionInfo.Normal(0, 1));
            GenerateHistogram(rnd, "gpurandom.normal(0, 1)");
        }

        [TestCase()]
        public void RandomGPU_henormal()
        {
            using GPURandom rnd = new GPURandom(RandomDistributionInfo.HeNormalFromSize(0, 1, samplecount));
            GenerateHistogram(rnd, $"gpurandom.henormal(0, 1, {samplecount})");
        }
        
        //[TestCase()]
        //public void RandomGPU_glorot()
        //{
        //    using GPURandom rnd = new GPURandom(RandomDistributionInfo.HeNormalFromSize(0, 1, samplecount));
        //    GenerateHistogram(rnd, $"gpurandom.henormal(0, 1, {samplecount})");
        //}

        [TestCase()]
        public void RandomGPU_lognormal()
        {
            using GPURandom rnd = new GPURandom(RandomDistributionInfo.LogNormal(0, 1));
            GenerateHistogram(rnd, $"gpurandom.gaussian(0, 1)");
        }

        [TestCase()]
        public void RandomGPU_uniform()
        {
            using GPURandom rnd = new GPURandom(RandomDistributionInfo.Uniform(0, 1));
            GenerateHistogram(rnd, $"gpurandom.uniform(0, 1)");
        }

        [TestCase()]
        public void RandomCPU_uniform()
        {
            CPURandom rnd = new CPURandom(RandomDistributionInfo.Uniform(0, 1));
            GenerateHistogram(rnd, $"cpurandom.uniform(0, 1)");
        }

        [TestCase()]
        public void RandomCPU_uniform01()
        {
            CPURandom rnd = new CPURandom(RandomDistributionInfo.Uniform(0.5f, 0.5f));
            GenerateHistogram(rnd, $"cpurandom.uniform(0.5, 0.5)");
        }

        [TestCase()]
        public void RandomCPU_normal()
        {
            CPURandom rnd = new CPURandom(RandomDistributionInfo.Normal(0, 1));
            GenerateHistogram(rnd, $"cpurandom.normal(0, 1)");
        }


        [TestCase()]
        public void RandomCPU_lognormal()
        {
            CPURandom rnd = new CPURandom(RandomDistributionInfo.LogNormal(0, 1));
            GenerateHistogram(rnd, $"cpurandom.lognormal(0, 1)");
        }

        [TestCase()]
        public void RandomCPU_henormal()
        {
            CPURandom rnd = new CPURandom(RandomDistributionInfo.HeNormalFromSize(0, 1, samplecount));
            GenerateHistogram(rnd, $"cpurandom.henormal(0, 1, {samplecount})");
        }

        [TestCase()]
        public void RandomCPU_glorotnormal()
        {
            CPURandom rnd = new CPURandom(RandomDistributionInfo.GlorotFromSize(0, 1, samplecount, samplecount));
            GenerateHistogram(rnd, $"cpurandom.glorot(0, 1, {samplecount}, {samplecount})");
        }

        public void GenerateHistogram(IRandom rnd, string title)
        { 
            Histogram h = new Histogram(0, 1, 4, 12); 
            for(int i = 0; i < samplecount; i++)
            {
                h.AddSample(rnd.NextSingle());
            }

            Bitmap bmp = h.GenerateBitmap(800, 800, title);
            string filename = Path.GetTempFileName() + ".bmp";
            bmp.Save(filename); 
//            System.Diagnostics.Process.Start(filename,);


            Process p = new Process();
            p.StartInfo = new ProcessStartInfo()
            {
                UseShellExecute = true
                , FileName = filename
            };
            p.Start(); 

        }


    }
}
                                            