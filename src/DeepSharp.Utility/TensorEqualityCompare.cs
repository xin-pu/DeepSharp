namespace DeepSharp.Utility
{
    public class TensorEqualityCompare : IEqualityComparer<torch.Tensor>
    {
        public bool Equals(torch.Tensor? x, torch.Tensor? y)
        {
            return x!.Equals(y!);
        }

        public int GetHashCode(torch.Tensor obj)
        {
            return -1;
        }
    }
}
