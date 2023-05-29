namespace DeepSharp.Dataset
{
    /// <summary>
    ///     基于异步流实现无穷尽的循环loader
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class InfiniteDataLoader<T> : DataLoader<T>
        where T : DataView

    {
        public InfiniteDataLoader(Dataset<T> dataset, DataLoaderConfig config) : base(dataset, config)
        {
            IEnumerator = GetEnumerator();
        }

        public IEnumerator<DataViewPair> IEnumerator { protected set; get; }

        public async IAsyncEnumerable<DataViewPair> GetBatchSample(int sample)
        {
            var i = 0;
            while (i++ < sample)
                if (IEnumerator.MoveNext())
                {
                    yield return IEnumerator.Current;
                }
                else
                {
                    IEnumerator.Reset();
                    IEnumerator.MoveNext();
                    yield return IEnumerator.Current;
                }

            await Task.CompletedTask;
        }
    }
}