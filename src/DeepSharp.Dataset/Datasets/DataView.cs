namespace DeepSharp.Dataset
{
    public abstract class DataView
    {
        /// <summary>
        ///     Get the features with tensor format.
        /// </summary>
        /// <returns></returns>
        public abstract torch.Tensor GetFeatures();

        /// <summary>
        ///     Get the labels with tensor format.
        /// </summary>
        /// <returns></returns>
        public abstract torch.Tensor GetLabels();


        /// <summary>
        ///     convert batch DataView to  single DataView Pair
        /// </summary>
        /// <param name="datasetViews"></param>
        /// <param name="device">cpu or  cuda</param>
        /// <returns></returns>
        public static DataViewPair FromDataViews(IEnumerable<DataView> datasetViews, torch.Device device)
        {
            var views = datasetViews.ToList();
            var features = views.Select(a => a.GetFeatures()).ToList();
            var labels = views.Select(a => a.GetLabels()).ToList();
            var result = new DataViewPair(labels, features).To(device);
            return result;
        }
    }
}