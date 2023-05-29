using DeepSharp.Dataset;
using DeepSharp.Dataset.Models;
using TorchSharpTest.SampleDataset;

namespace TorchSharpTest.TorchTests
{
    public class DataSetTest : AbstractTest
    {
        public DataSetTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void StreamDatasetTest()
        {
            var dataset = new ObjectDataset<IrisData>(@"F:\Iris\iris-train.txt");
            var res = dataset.GetTensor(0);
            Print(res);
        }

        [Fact]
        public void StreamDataloaderTest()
        {
            var dataset = new ObjectDataset<IrisData>(@"F:\Iris\iris-train.txt");
            var dataloader =
                new torch.utils.data.DataLoader<IrisData, DataViewPair>(dataset, 4,
                    DataViewPair.FromDataViews, true, new torch.Device(DeviceType.CUDA));

            using var iterator = dataloader.GetEnumerator();
            while (iterator.MoveNext())
            {
                var current = iterator.Current;
                Print(current);
            }
        }
    }
}