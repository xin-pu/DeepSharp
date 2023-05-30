namespace DeepSharp.Dataset
{
    /// <summary>
    ///     Basic DataLoader inherit from torch.utils.data.DataLoader
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class DataLoader<T> : torch.utils.data.DataLoader<T, DataViewPair>
        where T : DataView
    {
        public DataLoader(Dataset<T> dataset, DataLoaderConfig config)
            : base(dataset, config.BatchSize, CollateFunc, config.Shuffle, config.Device, config.Seed, config.NumWorker,
                config.DropLast)
        {
        }

        public static DataViewPair CollateFunc(IEnumerable<DataView> dataViews, torch.Device device)
        {
            var views = dataViews.ToList();
            var features = views.Select(a => a.GetFeatures()).ToList();
            var labels = views.Select(a => a.GetLabels()).ToList();
            var result = new DataViewPair(labels, features).To(device);
            return result;
        }

        public async IAsyncEnumerable<DataViewPair> GetBatchSample()
        {
            using var enumerator = GetEnumerator();
            while (enumerator.MoveNext()) yield return enumerator.Current;
            await Task.CompletedTask;
        }
    }
}