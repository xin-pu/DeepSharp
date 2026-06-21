using DeepSharp.Dataset;
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
			var dataset = new Dataset<Iris>(@"F:\Iris\iris-train.txt");
			var res     = dataset.GetTensor(0);
			Print(res);
		}

		[Fact]
		public void OriginalDataloaderTest()
		{
			var dataset = new Dataset<Iris>(@"F:\Iris\iris-train.txt");
			var device  = new Device(DeviceType.CUDA);
			var dataloader =
				new DataLoader<Iris>(dataset, new DataLoaderConfig { BatchSize = 4, Shuffle = true, Device = device });

			using var iterator = dataloader.GetEnumerator();
			while (iterator.MoveNext())
			{
				var current = iterator.Current;
				Print(current);
			}
		}

		[Fact]
		public void DataLoaderTest()
		{
			var dataset = new Dataset<Iris>(@"F:\Iris\iris-train.txt");
			var dataConfig = new DataLoaderConfig
			{
				Device = new Device(DeviceType.CUDA)
			};
			var dataloader = new DataLoader<Iris>(dataset, dataConfig);

			using var iterator = dataloader.GetEnumerator();
			while (iterator.MoveNext())
			{
				var current = iterator.Current;
				Print(current);
			}
		}

		[Fact]
		public async void InfiniteDataLoaderTest()
		{
			var dataset    = new Dataset<Iris>(@"F:\Iris\iris-train.txt");
			var dataConfig = new DataLoaderConfig();
			var dataloader = new InfiniteDataLoader<Iris>(dataset, dataConfig);

			await foreach (var a in dataloader.GetBatchSample(100))
			{
				var array = a.Labels.data<float>().ToArray();
				Print($"{string.Join(";", array)}");
			}
		}
	}
}