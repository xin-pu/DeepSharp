using DeepSharp.Dataset;
using DeepSharp.Dataset.Datasets;

namespace TorchSharpTest.SampleDataset
{
    public class Iris : DataView
    {
        /// <summary>
        /// </summary>
        public Iris()
        {
        }

        [StreamHeader(0)] public long Label { set; get; }
        [StreamHeader(1)] public float SepalLength { set; get; }
        [StreamHeader(2)] public float SepalWidth { set; get; }
        [StreamHeader(3)] public float PetalLength { set; get; }
        [StreamHeader(4)] public float PetalWidth { set; get; }

        public override torch.Tensor GetFeatures()
        {
            return torch.tensor(new[] {SepalLength, SepalWidth, PetalLength, PetalWidth});
        }

        public override torch.Tensor GetLabels()
        {
            return torch.tensor(new[] {Label});
        }

        public override string ToString()
        {
            return $"Label:{Label}\t" +
                   $"SepalLength:{SepalLength:F2}\tSepalWidth:{SepalWidth:F2}\t" +
                   $"PetalLength:{PetalLength:F2}\tPetalWidth:{PetalWidth:F2}";
        }

        /// <summary>
        ///     return a random Iris
        /// </summary>
        /// <returns></returns>
        public static Iris RandomIris()
        {
            var randomSource = new Random();
            return new Iris
            {
                Label = randomSource.Next(0, 3),
                PetalLength = randomSource.NextSingle() * 4,
                PetalWidth = randomSource.NextSingle() * 4,
                SepalLength = randomSource.NextSingle() * 4,
                SepalWidth = randomSource.NextSingle() * 4
            };
        }
    }
}