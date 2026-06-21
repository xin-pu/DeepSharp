namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     Step
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
		///     动作
		/// </summary>
		public Act Action { get; set; }

		/// <summary>
		///     动作后的观察
		/// </summary>
		public Observation PostState { get; set; }

		/// <summary>
		///     动作后的奖励
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