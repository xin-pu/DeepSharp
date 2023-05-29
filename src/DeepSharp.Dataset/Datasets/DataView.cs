namespace DeepSharp.Dataset
{
    public abstract class DataView
    {
        /// <summary>
        ///     转换为单个数据的特征张量，不含批次信息
        /// </summary>
        /// <returns></returns>
        public abstract torch.Tensor GetFeatures();

        /// <summary>
        ///     转换为单个数据的标签张量，不含批次信息
        /// </summary>
        /// <returns></returns>
        public abstract torch.Tensor GetLabels();


        /// <summary>
        ///     将DataView对象转为 最终训练用的 特征与标签 张量 对象
        /// </summary>
        /// <param name="datasetViews"></param>
        /// <param name="device">训练时使用的设备</param>
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