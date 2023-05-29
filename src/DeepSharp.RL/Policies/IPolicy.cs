namespace DeepSharp.RL.Policies
{
    public interface IPolicy
    {
        public torch.Tensor Predict(torch.Tensor input);
    }
}