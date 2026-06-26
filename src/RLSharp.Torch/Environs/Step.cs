namespace RLSharp.Torch.Environs
{
	/// <summary>
	///     A single time step containing pre-state, action, post-state, and reward.
	/// </summary>
	public class Step : ICloneable
	{
		public Step(ObservationValue preState,
			ActionValue              action,
			ObservationValue         postState,
			Reward                   reward,
			bool                     isComplete = false,
			float                    priority   = 1f)
		{
			PreState   = preState;
			Action     = action;
			Reward     = reward;
			PostState  = postState;
			IsComplete = isComplete;
			Priority   = priority;
		}

		public ObservationValue PreState { get; set; }

		/// <summary>
		///     Action performed by the agent.
		/// </summary>
		public ActionValue Action { get; set; }

		/// <summary>
		///     ObservationValue after the action.
		/// </summary>
		public ObservationValue PostState { get; set; }

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