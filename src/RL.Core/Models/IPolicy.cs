namespace RL.Core
{
    public interface IPolicy
    {
        public Tensor Predict(Tensor input);
    }
}