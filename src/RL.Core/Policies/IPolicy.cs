namespace RL.Core.Policies
{
    public interface IPolicy
    {
        public Tensor Predict(Tensor input);
    }
}