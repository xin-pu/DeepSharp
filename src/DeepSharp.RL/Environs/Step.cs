namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     Step
    /// </summary>
    public class Step : ICloneable
    {
        public Step(Observation preState,
            Act action,
            Observation postState,
            Reward reward,
            bool isComplete = false,
            float priority = 1f)
        {
            PreState = preState;
            Action = action;
            Reward = reward;
            PostState = postState;
            IsComplete = isComplete;
            Priority = priority;
        }

        public Observation PreState { set; get; }

        /// <summary>
        ///     动作
        /// </summary>
        public Act Action { set; get; }

        /// <summary>
        ///     动作后的观察
        /// </summary>
        public Observation PostState { set; get; }

        /// <summary>
        ///     动作后的奖励
        /// </summary>
        public Reward Reward { set; get; }

        public bool IsComplete { set; get; }

        public float Priority { set; get; }


        public object Clone()
        {
            return new Step(PreState, Action, PostState, Reward, IsComplete);
        }
    }
}