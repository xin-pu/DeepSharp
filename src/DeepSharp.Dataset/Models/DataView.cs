namespace DeepSharp.Dataset.Models
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
    }
}