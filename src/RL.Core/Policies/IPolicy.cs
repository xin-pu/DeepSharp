namespace RL.Core.Policies
{
    public interface IPolicy
    {
        public torch.Tensor Predict(torch.Tensor input);
    }
}