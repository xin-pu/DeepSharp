namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     Step
    /// </summary>
    public class Step
    {
        public Step(Act action, Observation observation, Reward reward, bool isComplete = false)
        {
            Action = action;
            Observation = observation;
            Reward = reward;
            IsComplete = isComplete;
        }

        /// <summary>
        ///     动作
        /// </summary>
        public Act Action { set; get; }

        /// <summary>
        ///     动作后的观察
        /// </summary>
        public Observation Observation { set; get; }

        /// <summary>
        ///     动作后的奖励
        /// </summary>
        public Reward Reward { set; get; }

        public bool IsComplete { set; get; }
    }
}