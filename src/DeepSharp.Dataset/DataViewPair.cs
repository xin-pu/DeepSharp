using DeepSharp.Dataset.Datasets;

namespace DeepSharp.Dataset
{
    /// <summary>
    ///     包含批次信息的 数据集 张量对象
    /// </summary>
    public class DataViewPair
    {
        /// <summary>
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="features"></param>
        public DataViewPair(torch.Tensor labels, torch.Tensor features)
        {
            Labels = labels;
            Features = features;
        }

        public DataViewPair(IEnumerable<torch.Tensor> labels, IEnumerable<torch.Tensor> features)
        {
            var labelsArray = labels.ToArray();
            var featuresArray = features.ToArray();
            Labels = torch.vstack(labelsArray);
            Features = torch.vstack(featuresArray);
        }


        public torch.Tensor Labels { set; get; }
        public torch.Tensor Features { set; get; }

        public override string ToString()
        {
            var strbuild = new StringBuilder();
            strbuild.AppendLine($"{Labels}");
            strbuild.AppendLine($"{Features}");
            return strbuild.ToString();
        }


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
            var result = new DataViewPair(labels, features);
            return result;
        }
    }
}