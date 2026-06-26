namespace RLSharp.Torch.Enumerates
{
	/// <summary>
	///     Play mode of the agent.
	/// </summary>
	public enum PlayMode
	{
		/// <summary>
		///     Random sampling from the action space.
		/// </summary>
		Sample,

		/// <summary>
		///     Actions selected by the agent's policy.
		/// </summary>
		Agent,

		/// <summary>
		///     Sample with probability ¦Å, agent policy with probability 1-¦Å.
		/// </summary>
		EpsilonGreedy
	}
}