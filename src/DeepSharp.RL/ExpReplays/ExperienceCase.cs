namespace DeepSharp.RL.ExperienceSources
{
	/// <summary>
	/// </summary>
	public struct ExperienceCase
	{
		public ExperienceCase(torch.Tensor preState,
			torch.Tensor                   action,
			torch.Tensor                   reward,
			torch.Tensor                   postState,
			torch.Tensor                   done)
		{
			PreState  = preState;
			Action    = action;
			Reward    = reward;
			PostState = postState;
			Done      = done;
		}

		/// <summary>
		///     State before action
		/// </summary>
		public torch.Tensor PreState { get; set; }

		/// <summary>
		///     Action
		/// </summary>
		public torch.Tensor Action { get; set; }

		/// <summary>
		///     Reward
		/// </summary>
		public torch.Tensor Reward { get; set; }

		/// <summary>
		///     State after action
		/// </summary>
		public torch.Tensor PostState { get; set; }

		/// <summary>
		///     Episode is complete?
		/// </summary>
		public torch.Tensor Done { get; set; }
	}
}