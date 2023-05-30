namespace DeepSharp.Dataset
{
    /// <summary>
    ///     Infinite DataLoader inherit from DataLoader
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class InfiniteDataLoader<T> : DataLoader<T>
        where T : DataView

    {
        public InfiniteDataLoader(Dataset<T> dataset, DataLoaderConfig config) : base(dataset, config)
        {
            IEnumerator = GetEnumerator();
        }

        protected IEnumerator<DataViewPair> IEnumerator { set; get; }

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