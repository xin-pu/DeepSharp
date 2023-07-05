namespace DeepSharp.RL.ExperienceSources
{
    /// <summary>
    /// </summary>
    public struct ExperienceCase
    {
        public ExperienceCase(torch.Tensor preState,
            torch.Tensor action,
            torch.Tensor reward,
            torch.Tensor postState,
            torch.Tensor done)
        {
            PreState = preState;
            Action = action;
            Reward = reward;
            PostState = postState;
            Done = done;
        }

        /// <summary>
        ///     State before action
        /// </summary>
        public torch.Tensor PreState { get; set; }

        /// <summary>
        ///     Action
        /// </summary>
        public torch.Tensor Action { set; get; }

        /// <summary>
        ///     Reward
        /// </summary>
        public torch.Tensor Reward { set; get; }

        /// <summary>
        ///     State after action
        /// </summary>
        public torch.Tensor PostState { set; get; }

        /// <summary>
        ///     Episode is complete?
        /// </summary>
        public torch.Tensor Done { set; get; }
    }
}