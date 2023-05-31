namespace DeepSharp.RL.Models
{
    /// <summary>
    ///     Step
    /// </summary>
    public struct Step
    {
        /// <summary>
        ///     动作
        /// </summary>
        public Action Action { set; get; }

        /// <summary>
        ///     动作后的观察
        /// </summary>
        public Observation Observation { set; get; }

        /// <summary>
        ///     动作后的奖励
        /// </summary>
        public Reward Reward { set; get; }
    }
}