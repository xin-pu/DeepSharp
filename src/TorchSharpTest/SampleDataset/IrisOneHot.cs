namespace TorchSharpTest.SampleDataset
{
    public class IrisOneHot : Iris
    {
        /// <summary>
        ///     OneHot [0,0,1] 代表分类3
        /// </summary>
        /// <returns></returns>
        public override torch.Tensor GetLabels()
        {
            var array = Enumerable.Repeat(0, 3).Select(a => (float) a).ToArray();
            array[Label] = 1;
            return torch.tensor(array);
        }
    }
}