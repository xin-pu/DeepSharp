using System.Collections;

namespace DeepSharp.Dataset
{
	/// <summary>
	///     Custom DataLoader implementing batching, shuffling, and device transfer.
	///     (torch.utils.data.DataLoader removed in TorchSharp 0.107.0)
	/// </summary>
	/// <typeparam name="T">DataView type</typeparam>
	public class DataLoader<T> : IEnumerable<DataViewPair>
		where T : DataView
	{
		public DataLoader(Dataset<T> dataset, DataLoaderConfig config)
		{
			Dataset    = dataset;
			Config     = config;
			BatchCount = (int)Math.Ceiling((double)dataset.Count / config.BatchSize);
		}

		public Dataset<T> Dataset { get; }

		public DataLoaderConfig Config { get; }

		public int BatchCount { get; }

		public IEnumerator<DataViewPair> GetEnumerator()
		{
			var indices = Enumerable.Range(0, (int)Dataset.Count).ToArray();

			if (Config.Shuffle)
			{
				var rng = Config.Seed is { } seed ? new Random(seed) : new Random();
				for (var i = indices.Length - 1; i > 0; i--)
				{
					var j = rng.Next(i + 1);
					(indices[i], indices[j]) = (indices[j], indices[i]);
				}
			}

			for (var i = 0; i < indices.Length; i += Config.BatchSize)
			{
				var batchSize = Math.Min(Config.BatchSize, indices.Length - i);

				if (Config.DropLast && batchSize < Config.BatchSize)
					yield break;

				var batch = indices[i..(i + batchSize)]
					.Select(idx => (DataView)Dataset.GetTensor(idx))
					.ToList();

				yield return DataView.FromDataViews(batch, Config.Device);
			}
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			return GetEnumerator();
		}

		public async IAsyncEnumerable<DataViewPair> GetBatchSample()
		{
			using var enumerator = GetEnumerator();
			while (enumerator.MoveNext()) yield return enumerator.Current;
			await Task.CompletedTask;
		}
	}
}