namespace DeepSharp.RL.ActionSelectors
{
    /// <summary>
    ///     Action Selector help to change Pred-out from Net to Specific action objects
    ///     动作选择器，转换网络的输出到具体的动作选择器
    /// </summary>
    public abstract class ActionSelector
    {
        protected ActionSelector(bool keepDims = false)
        {
            KeepDims = keepDims;
        }

        public bool KeepDims { set; get; }

        public abstract torch.Tensor Select(torch.Tensor probs);
    }
}