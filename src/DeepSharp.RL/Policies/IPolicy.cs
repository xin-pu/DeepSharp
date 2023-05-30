namespace DeepSharp.RL.Policies
{
    public interface IPolicy
    {
        /// <summary>
        ///     get next action according observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public torch.Tensor PredictAction(torch.Tensor observation);
    }
}