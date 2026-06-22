namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     A single time step containing pre-state, action, post-state, and reward.
	/// </summary>
	public class Step : ICloneable
	{
		public Step(Observation preState,
			Act                 action,
			Observation         postState,
			Reward              reward,
			bool                isComplete = false,
			float               priority   = 1f)
		{
			PreState   = preState;
			Action     = action;
			Reward     = reward;
			PostState  = postState;
			IsComplete = isComplete;
			Priority   = priority;
		}

		public Observation PreState { get; set; }

		/// <summary>
		///     Action performed by the agent.
		/// </summary>
		public Act Action { get; set; }

		/// <summary>
		///     Observation after the action.
		/// </summary>
		public Observation PostState { get; set; }

		/// <summary>
		///     Reward received after the action.
		/// </summary>
		public Reward Reward { get; set; }

		public bool IsComplete { get; set; }

		public float Priority { get; set; }


		public object Clone()
		{
			return new Step(PreState, Action, PostState, Reward, IsComplete);
		}
	}
}