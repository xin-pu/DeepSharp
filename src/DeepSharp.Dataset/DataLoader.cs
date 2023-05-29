namespace DeepSharp.Dataset
{
    public class DataLoader<T> : torch.utils.data.DataLoader<T, DataViewPair> where T : DataView
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
    }
}