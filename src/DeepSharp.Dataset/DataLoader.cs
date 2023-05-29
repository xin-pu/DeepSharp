namespace DeepSharp.Dataset
{
    public class DataLoader<T> : torch.utils.data.DataLoader<T, DataViewPair> where T : DataView
    {
        public DataLoader(Dataset<T> dataset, DataLoaderConfig config, torch.Device device)
            : base(dataset, config.BatchSize, CollateFunc, config.Shuffle, device, config.Seed, config.NumWorker,
                config.DropLast)
        {
        }

        public static DataViewPair CollateFunc(IEnumerable<DataView> dataViews, torch.Device device)
        {
            var views = dataViews.ToList();
            var features = views.Select(a => a.GetFeatures()).ToList();
            var labels = views.Select(a => a.GetLabels()).ToList();
            var result = new DataViewPair(labels, features);
            return result;
        }
    }
}