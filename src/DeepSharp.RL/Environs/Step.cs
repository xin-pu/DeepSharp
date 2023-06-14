namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     Step
    /// </summary>
    public class Step
    {
        public Step(Observation state, Act action, Observation stateNew, Reward reward, bool isComplete = false)
        {
            State = state;
            Action = action;
            Reward = reward;
            StateNew = stateNew;
            IsComplete = isComplete;
        }

        public Observation State { set; get; }

        /// <summary>
        ///     动作
        /// </summary>
        public Act Action { set; get; }

        /// <summary>
        ///     动作后的观察
        /// </summary>
        public Observation StateNew { set; get; }

        /// <summary>
        ///     动作后的奖励
        /// </summary>
        public Reward Reward { set; get; }

        public bool IsComplete { set; get; }
    }
}