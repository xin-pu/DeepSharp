namespace DeepSharp.RL.Enumerates
{
    /// <summary>
    ///     Play mode of Agent
    /// </summary>
    public enum PlayMode
    {
        /// <summary>
        ///     平均采样
        /// </summary>
        Sample,

        /// <summary>
        ///     根据智能体的策略
        /// </summary>
        Agent,

        /// <summary>
        ///     Sample(ε) and Agent(1-ε)
        /// </summary>
        EpsilonGreedy
    }
}