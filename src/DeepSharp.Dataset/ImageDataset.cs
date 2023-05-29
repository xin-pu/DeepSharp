namespace DeepSharp.Dataset
{
    public class ImageDataset<T> : torch.utils.data.Dataset<T>
    {
        public override long Count { get; }

        public override T GetTensor(long index)
        {
            throw new NotImplementedException();
        }
    }
}